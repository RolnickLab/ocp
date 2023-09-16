from math import pi as PI
from math import sqrt

import torch
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    Envelope,
    ResidualLayer,
    SphericalBasisLayer,
)
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc_inputs,
)
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.modules.pooling import Graclus, Hierarchical_Pooling
from ocpmodels.models.utils.activations import swish
from ocpmodels.models.afaenet import (
    GATInteraction,
    GaussianSmearing
)


try:
    import sympy as sym
except ImportError:
    sym = None

NUM_CLUSTERS = 20
NUM_POOLING_LAYERS = 1


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super().__init__()
        self.act = act

        self.emb = Embedding(85, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j, tags=None, subnodes=None):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class AdvancedEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        tag_hidden_channels,
        pg_hidden_channels,
        phys_hidden_channels,
        phys_embeds,
        graph_rewiring,
        act=swish,
    ):
        super().__init__()
        self.act = act
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0
        self.use_positional_embeds = graph_rewiring in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }
        # self.use_positional_embeds = False

        # Phys embeddings
        self.phys_emb = PhysEmbedding(props=phys_embeds, pg=self.use_pg)
        # With MLP
        if self.use_mlp_phys:
            self.phys_lin = Linear(self.phys_emb.n_properties, phys_hidden_channels)
        else:
            phys_hidden_channels = self.phys_emb.n_properties
        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = Embedding(
                self.phys_emb.period_size, pg_hidden_channels
            )
            self.group_embedding = Embedding(
                self.phys_emb.group_size, pg_hidden_channels
            )
        # Tag embedding
        if tag_hidden_channels:
            self.tag = Embedding(3, tag_hidden_channels)

        # Position encoding
        if self.use_positional_embeds:
            self.pe = PositionalEncoding(hidden_channels, 210)

        # Main embedding
        self.emb = Embedding(
            85,
            hidden_channels
            - tag_hidden_channels
            - phys_hidden_channels
            - 2 * pg_hidden_channels,
        )

        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        if self.use_mlp_phys:
            self.phys_lin.reset_parameters()
        if self.use_tag:
            self.tag.weight.data.uniform_(-sqrt(3), sqrt(3))
        if self.use_pg:
            self.period_embedding.weight.data.uniform_(-sqrt(3), sqrt(3))
            self.group_embedding.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j, tag=None, subnodes=None):
        x_ = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))

        if self.phys_emb.device != x.device:
            self.phys_emb = self.phys_emb.to(x.device)

        if self.use_tag:
            x_tag = self.tag(tag)
            x_ = torch.cat((x_, x_tag), dim=1)

        if self.phys_emb.n_properties > 0:
            x_phys = self.phys_emb.properties[x]
            if self.use_mlp_phys:
                x_phys = self.phys_lin(x_phys)
            x_ = torch.cat((x_, x_phys), dim=1)

        if self.use_pg:
            x_period = self.period_embedding(self.phys_emb.period[x])
            x_group = self.group_embedding(self.phys_emb.group[x])
            x_ = torch.cat((x_, x_period, x_group), dim=1)

        if self.use_positional_embeds:
            idx_of_non_zero_val = (tag == 0).nonzero().T.squeeze(0)
            x_pos = torch.zeros_like(x_, device=x_.device)
            x_pos[idx_of_non_zero_val, :] = self.pe(subnodes).to(device=x_pos.device)
            x_ += x_pos

        return self.act(
            self.lin(
                torch.cat(
                    [
                        x_[i],
                        x_[j],
                        rbf,
                    ],
                    dim=-1,
                )
            )
        )


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super(InteractionPPBlock, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations.
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets.
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection.
        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformations.
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis.
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis.
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings.
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class EHOutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        energy_head,
        act=swish,
    ):
        super(EHOutputPPBlock, self).__init__()
        self.act = act
        self.energy_head = energy_head

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        # weighted average & pooling
        if self.energy_head in {"pooling", "random"}:
            self.hierarchical_pooling = Hierarchical_Pooling(
                hidden_channels,
                self.act,
                NUM_POOLING_LAYERS,
                NUM_CLUSTERS,
                self.energy_head,
            )
        elif self.energy_head == "graclus":
            self.graclus = Graclus(hidden_channels, self.act)
        elif self.energy_head == "weighted-av-final-embeds":
            self.w_lin = Linear(hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)
        if self.energy_head == "weighted-av-final-embeds":
            self.w_lin.bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.w_lin.weight)

    def forward(self, x, rbf, i, edge_index, edge_weight, batch, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)

        pooling_loss = None
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(x)
        elif self.energy_head == "graclus":
            x, batch = self.graclus(x, edge_index, edge_weight, batch)
        elif self.energy_head in {"pooling", "random"}:
            x, batch, pooling_loss = self.hierarchical_pooling(
                x, edge_index, edge_weight, batch
            )

        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        x = self.lin(x)

        if self.energy_head == "weighted-av-final-embeds":
            x = x * alpha

        return x, pooling_loss, batch


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act=swish,
    ):
        super(OutputPPBlock, self).__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


@registry.register_model("adpp")
class ADPP(BaseModel):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.

    Args:
        hidden_channels (int): Hidden embedding size.
        tag_hidden_channels (int): tag embedding size
        pg_hidden_channels (int): period & group embedding size
        phys_hidden_channels (int): MLP hidden size for physics embedding
        phys_embeds (bool): whether we use physics embeddings or not
        graph_rewiring (str): name of rewiring method. Default=False.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        use_pbc (bool, optional): Use of periodic boundary conditions.
            (default: true)
        otf_graph (bool, optional): Recompute radius graph.
            (default: false)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation function.
            (default: :obj:`swish`)
        regress_forces: (bool, optional): Compute atom forces from energy.
            (default: false).
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained"

    def __init__(self, **kwargs):
        super().__init__()

        kwargs["num_targets"] = kwargs["hidden_channels"] // 2
        self.act = swish

        self.cutoff = kwargs["cutoff"]
        self.use_pbc = kwargs["use_pbc"]
        self.otf_graph = kwargs["otf_graph"]
        self.regress_forces = kwargs["regress_forces"]
        self.energy_head = kwargs["energy_head"]
        use_tag = kwargs["tag_hidden_channels"] > 0
        use_pg = kwargs["pg_hidden_channels"] > 0
        act = (
            getattr(nn.functional, kwargs["act"]) if kwargs["act"] != "swish" else swish
        )

        assert (
            kwargs["tag_hidden_channels"] + 2 * kwargs["pg_hidden_channels"] + 16
            < kwargs["hidden_channels"]
        )
        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        self.rbf_ads = BesselBasisLayer(
            kwargs["num_radial"], self.cutoff, kwargs["envelope_exponent"]
        )
        self.rbf_cat = BesselBasisLayer(
            kwargs["num_radial"], self.cutoff, kwargs["envelope_exponent"]
        )
        self.sbf_ads = SphericalBasisLayer(
            kwargs["num_spherical"],
            kwargs["num_radial"],
            self.cutoff,
            kwargs["envelope_exponent"],
        )
        self.sbf_cat = SphericalBasisLayer(
            kwargs["num_spherical"],
            kwargs["num_radial"],
            self.cutoff,
            kwargs["envelope_exponent"],
        )
        # Disconnected interaction embedding
        self.distance_expansion_disc = GaussianSmearing(
            0.0, 20.0, 100
        )
        self.disc_edge_embed = Linear(100, kwargs["hidden_channels"])

        if use_tag or use_pg or kwargs["phys_embeds"] or kwargs["graph_rewiring"]:
            self.emb_ads = AdvancedEmbeddingBlock(
                kwargs["num_radial"],
                kwargs["hidden_channels"],
                kwargs["tag_hidden_channels"],
                kwargs["pg_hidden_channels"],
                kwargs["phys_hidden_channels"],
                kwargs["phys_embeds"],
                kwargs["graph_rewiring"],
                act,
            )
            self.emb_cat = AdvancedEmbeddingBlock(
                kwargs["num_radial"],
                kwargs["hidden_channels"],
                kwargs["tag_hidden_channels"],
                kwargs["pg_hidden_channels"],
                kwargs["phys_hidden_channels"],
                kwargs["phys_embeds"],
                kwargs["graph_rewiring"],
                act,
            )
        else:
            self.emb_ads = EmbeddingBlock(
                kwargs["num_radial"], kwargs["hidden_channels"], act
            )
            self.emb_cat = EmbeddingBlock(
                kwargs["num_radial"], kwargs["hidden_channels"], act
            )

        if self.energy_head:
            self.output_blocks_ads = torch.nn.ModuleList(
                [
                    EHOutputPPBlock(
                        kwargs["num_radial"],
                        kwargs["hidden_channels"],
                        kwargs["out_emb_channels"],
                        kwargs["num_targets"],
                        kwargs["num_output_layers"],
                        self.energy_head,
                        act,
                    )
                    for _ in range(kwargs["num_blocks"] + 1)
                ]
            )
            self.output_blocks_cat = torch.nn.ModuleList(
                [
                    EHOutputPPBlock(
                        kwargs["num_radial"],
                        kwargs["hidden_channels"],
                        kwargs["out_emb_channels"],
                        kwargs["num_targets"],
                        kwargs["num_output_layers"],
                        self.energy_head,
                        act,
                    )
                    for _ in range(kwargs["num_blocks"] + 1)
                ]
            )
        else:
            self.output_blocks_ads = torch.nn.ModuleList(
                [
                    OutputPPBlock(
                        kwargs["num_radial"],
                        kwargs["hidden_channels"],
                        kwargs["out_emb_channels"],
                        kwargs["num_targets"],
                        kwargs["num_output_layers"],
                        act,
                    )
                    for _ in range(kwargs["num_blocks"] + 1)
                ]
            )
            self.output_blocks_cat = torch.nn.ModuleList(
                [
                    OutputPPBlock(
                        kwargs["num_radial"],
                        kwargs["hidden_channels"],
                        kwargs["out_emb_channels"],
                        kwargs["num_targets"],
                        kwargs["num_output_layers"],
                        act,
                    )
                    for _ in range(kwargs["num_blocks"] + 1)
                ]
            )

        self.interaction_blocks_ads = torch.nn.ModuleList(
            [
                InteractionPPBlock(
                    kwargs["hidden_channels"],
                    kwargs["int_emb_size"],
                    kwargs["basis_emb_size"],
                    kwargs["num_spherical"],
                    kwargs["num_radial"],
                    kwargs["num_before_skip"],
                    kwargs["num_after_skip"],
                    act,
                )
                for _ in range(kwargs["num_blocks"])
            ]
        )
        self.interaction_blocks_cat = torch.nn.ModuleList(
            [
                InteractionPPBlock(
                    kwargs["hidden_channels"],
                    kwargs["int_emb_size"],
                    kwargs["basis_emb_size"],
                    kwargs["num_spherical"],
                    kwargs["num_radial"],
                    kwargs["num_before_skip"],
                    kwargs["num_after_skip"],
                    act,
                )
                for _ in range(kwargs["num_blocks"])
            ]
        )
        self.inter_interactions = torch.nn.ModuleList(
            [
                GATInteraction(
                    kwargs["hidden_channels"],
                    kwargs["gat_mode"],
                    kwargs["hidden_channels"]
                )
                for _ in range(kwargs["num_blocks"])
            ]
        )

        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin_ads = Linear(kwargs["hidden_channels"], 1)
            self.w_lin_cat = Linear(kwargs["hidden_channels"], 1)

        self.task = kwargs["task_name"]

        self.combination = nn.Sequential(
            Linear(kwargs["hidden_channels"] // 2 * 2, kwargs["hidden_channels"] // 2),
            self.act,
            Linear(kwargs["hidden_channels"] // 2, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf_ads.reset_parameters()
        self.rbf_cat.reset_parameters()
        self.emb_ads.reset_parameters()
        self.emb_cat.reset_parameters()
        for out in self.output_blocks_ads:
            out.reset_parameters()
        for out in self.output_blocks_cat:
            out.reset_parameters()
        for interaction in self.interaction_blocks_ads:
            interaction.reset_parameters()
        for interaction in self.interaction_blocks_cat:
            interaction.reset_parameters()
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin_ads.bias.data.fill_(0)
            self.w_lin_cat.bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.w_lin.weight)

    def triplets(self, edge_index, cell_offsets, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()

        # Remove self-loop triplets d->b->d
        # Check atom as well as cell offset
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        (
            pos_ads,
            edge_index_ads,
            cell_ads,
            cell_offsets_ads,
            neighbors_ads,
            batch_ads,
            atomic_numbers_ads,
            tags_ads,
        ) = (
            data["adsorbate"].pos,
            data["adsorbate", "is_close", "adsorbate"].edge_index,
            data["adsorbate"].cell,
            data["adsorbate"].cell_offsets,
            data["adsorbate"].neighbors,
            data["adsorbate"].batch,
            data["adsorbate"].atomic_numbers,
            data["adsorbate"].tags,
        )
        (
            pos_cat,
            edge_index_cat,
            cell_cat,
            cell_offsets_cat,
            neighbors_cat,
            batch_cat,
            atomic_numbers_cat,
            tags_cat,
        ) = (
            data["catalyst"].pos,
            data["catalyst", "is_close", "catalyst"].edge_index,
            data["catalyst"].cell,
            data["catalyst"].cell_offsets,
            data["catalyst"].neighbors,
            data["catalyst"].batch,
            data["catalyst"].atomic_numbers,
            data["catalyst"].tags,
        )

        if self.otf_graph: # NOT IMPLEMENTED!!
            edge_index, cell_offsets, neighbors = radius_graph_pbc_inputs(
                pos,
                natoms,
                cell,
                self.cutoff,
                50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        # Rewire the graph
        subnodes = False

        if self.use_pbc:
            out = get_pbc_distances(
                pos_ads,
                edge_index_ads,
                cell_ads,
                cell_offsets_ads,
                neighbors_ads,
                return_offsets=True,
            )

            edge_index_ads = out["edge_index"]
            dist_ads = out["distances"]
            offsets_ads = out["offsets"]

            j_ads, i_ads = edge_index_ads

            out = get_pbc_distances(
                pos_cat,
                edge_index_cat,
                cell_cat,
                cell_offsets_cat,
                neighbors_cat,
                return_offsets=True,
            )

            edge_index_cat = out["edge_index"]
            dist_cat = out["distances"]
            offsets_cat = out["offsets"]

            j_cat, i_cat = edge_index_cat
        else: # NOT IMPLEMENTED
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            j, i = edge_index
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        _, _, idx_i_ads, idx_j_ads, idx_k_ads, idx_kj_ads, idx_ji_ads = self.triplets(
            edge_index_ads,
            cell_offsets_ads,
            num_nodes=atomic_numbers_ads.size(0),
        )
        _, _, idx_i_cat, idx_j_cat, idx_k_cat, idx_kj_cat, idx_ji_cat = self.triplets(
            edge_index_cat,
            cell_offsets_cat,
            num_nodes=atomic_numbers_cat.size(0),
        )

        # Calculate angles.
        pos_i_ads = pos_ads[idx_i_ads].detach()
        pos_j_ads = pos_ads[idx_j_ads].detach()

        pos_i_cat = pos_cat[idx_i_cat].detach()
        pos_j_cat = pos_cat[idx_j_cat].detach()
        if self.use_pbc:
            pos_ji_ads, pos_kj_ads = (
                pos_ads[idx_j_ads].detach() - pos_i_ads + offsets_ads[idx_ji_ads],
                pos_ads[idx_k_ads].detach() - pos_j_ads + offsets_ads[idx_kj_ads],
            )
            pos_ji_cat, pos_kj_cat = (
                pos_cat[idx_j_cat].detach() - pos_i_cat + offsets_cat[idx_ji_cat],
                pos_cat[idx_k_cat].detach() - pos_j_cat + offsets_cat[idx_kj_cat],
            )
        else: # NOT IMPLEMENTED
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i,
                pos[idx_k].detach() - pos_j,
            )

        a_ads = (pos_ji_ads * pos_kj_ads).sum(dim=-1)
        b_ads = torch.cross(pos_ji_ads, pos_kj_ads).norm(dim=-1)
        angle_ads = torch.atan2(b_ads, a_ads)

        a_cat = (pos_ji_cat * pos_kj_cat).sum(dim=-1)
        b_cat = torch.cross(pos_ji_cat, pos_kj_cat).norm(dim=-1)
        angle_cat = torch.atan2(b_cat, a_cat)

        rbf_ads = self.rbf_ads(dist_ads)
        sbf_ads = self.sbf_ads(dist_ads, angle_ads, idx_kj_ads)

        rbf_cat = self.rbf_cat(dist_cat)
        sbf_cat = self.sbf_cat(dist_cat, angle_cat, idx_kj_cat)

        pooling_loss = None  # deal with pooling loss

        # Embedding block.
        x_ads = self.emb_ads(atomic_numbers_ads.long(), rbf_ads, i_ads, j_ads, tags_ads, subnodes)
        if self.energy_head:
            P_ads, pooling_loss, batch_ads = self.output_blocks_ads[0](
                x_ads, rbf_ads, i_ads, edge_index_ads, dist_ads, batch_ads, num_nodes=pos_ads.size(0)
            )
        else:
            P_ads = self.output_blocks_ads[0](x_ads, rbf_ads, i_ads, num_nodes=pos_ads.size(0))

        if self.energy_head == "weighted-av-initial-embeds":
            alpha_ads = self.w_lin_ads(scatter(x_ads, i_ads, dim=0, dim_size=pos_ads.size(0)))

        x_cat = self.emb_cat(atomic_numbers_cat.long(), rbf_cat, i_cat, j_cat, tags_cat, subnodes)
        if self.energy_head:
            P_cat, pooling_loss, batch_cat = self.output_blocks_cat[0](
                x_cat, rbf_cat, i_cat, edge_index_cat, dist_cat, batch_cat, num_nodes=pos_cat.size(0)
            )
        else:
            P_cat = self.output_blocks_cat[0](x_cat, rbf_cat, i_cat, num_nodes=pos_cat.size(0))

        if self.energy_head == "weighted-av-initial-embeds":
            alpha_cat = self.w_lin_cat(scatter(x_cat, i_cat, dim=0, dim_size=pos_cat.size(0)))

        edge_weights = self.distance_expansion_disc(data["is_disc"].edge_weight)
        edge_weights = self.disc_edge_embed(edge_weights)

        # Interaction blocks.
        energy_Ps_ads = []
        energy_Ps_cat = []

        for (
            interaction_block_ads,
            interaction_block_cat,
            output_block_ads,
            output_block_cat,
            disc_interaction,
        ) in zip(
            self.interaction_blocks_ads, 
            self.interaction_blocks_cat,
            self.output_blocks_ads[1:],
            self.output_blocks_cat[1:],
            self.inter_interactions,
        ):
            intra_ads = interaction_block_ads(x_ads, rbf_ads, sbf_ads, idx_kj_ads, idx_ji_ads)
            intra_cat = interaction_block_cat(x_cat, rbf_cat, sbf_cat, idx_kj_cat, idx_ji_cat)

            inter_ads, inter_cat = disc_interaction(
                intra_ads,
                intra_cat,
                data["is_disc"].edge_index,
                edge_weights
            )

            x_ads, x_cat = x_ads + inter_ads, x_cat + inter_cat
            x_ads, x_cat = nn.functional.normalize(x_ads), nn.functional.normalize(x_cat)

            if self.energy_head:
                P_bis_ads, pooling_loss_bis_ads, _ = output_block_ads(
                    x_ads, rbf_ads, i_ads, edge_index_ads, dist_ads, batch_ads, num_nodes=pos_ads.size(0)
                )
                energy_Ps_ads.append(
                    P_bis_ads.sum(0) / len(P)
                    if batch_ads is None
                    else scatter(P_bis_ads, batch_ads, dim=0)
                )
                if pooling_loss_bis_ads is not None:
                    pooling_loss += pooling_loss_bis_ads

                P_bis_cat, pooling_loss_bis_cat, _ = output_block_cat(
                    x_cat, rbf_cat, i_cat, edge_index_cat, dist_cat, batch_cat, num_nodes=pos_cat.size(0)
                )
                energy_Ps_cat.append(
                    P_bis_cat.sum(0) / len(P)
                    if batch_cat is None
                    else scatter(P_bis_cat, batch_cat, dim=0)
                )
                if pooling_loss_bis_cat is not None:
                    pooling_loss += pooling_loss_bis_cat
            else:
                P_ads += output_block_ads(x_ads, rbf_ads, i_ads, num_nodes=pos_ads.size(0))
                P_cat += output_block_cat(x_cat, rbf_cat, i_cat, num_nodes=pos_cat.size(0))

        if self.energy_head == "weighted-av-initial-embeds":
            P = P * alpha

        import ipdb
        ipdb.set_trace()

        # Output
        # scatter
        energy_ads = self.scattering(batch_ads, P_ads)
        energy_cat = self.scattering(batch_cat, P_cat)
        energy = torch.cat([energy_ads, energy_cat], dim=1)
        energy = self.combination(energy)

        return {
            "energy": energy,
            "pooling_loss": pooling_loss,
        }

    def scattering(self, batch, P, P_bis=0):
        energy = scatter(P, batch, dim=0, reduce="add")

        return energy

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        return

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
