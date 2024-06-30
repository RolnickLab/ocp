"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the DimeNet implementation as part of
pytorch-geometric: https://github.com/rusty1s/pytorch_geometric. License:

---

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from math import pi as PI
from math import sqrt

import torch
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock
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
    radius_graph_pbc,
)
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.force_decoder import ForceDecoder
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.models.gemnet.layers.embedding_block import AtomEmbedding
from ocpmodels.models.gemnet.layers.base_layers import Dense
from ocpmodels.models.utils.activations import swish
from ocpmodels.modules.ewald_block import (
    EwaldBlock,
    x_to_k_cell,
    get_k_index_product_set,
    get_k_voxel_grid,
    x_to_k_cell,
)

try:
    import sympy as sym
except ImportError:
    sym = None


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
        use_ewald=False,
    ):
        super(EHOutputPPBlock, self).__init__()
        self.act = act
        self.energy_head = energy_head
        self.use_ewald = use_ewald

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()

        if self.use_ewald:
            # Combine Ewald-updated atom encodings with the remaining hidden state
            # of the model by the concatenating Ewald and short-range embeddings
            # before the linear upprojection layer of each output block
            self.lin_up = nn.Linear(2 * hidden_channels, out_emb_channels, bias=True)

        else:
            self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        # weighted average
        if self.energy_head == "weighted-av-final-embeds":
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

    def forward(
        self, x, rbf, i, edge_index, edge_weight, batch, num_nodes=None, h=None
    ):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)

        if self.use_ewald:
            x = torch.cat([x, h], dim=-1)

        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(x)

        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        x = self.lin(x)

        if self.energy_head == "weighted-av-final-embeds":
            x = x * alpha

        return x, batch


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act=swish,
        use_ewald=False,
    ):
        super(OutputPPBlock, self).__init__()
        self.act = act
        self.use_ewald = use_ewald

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        if self.use_ewald:
            # Combine Ewald-updated atom encodings with the remaining hidden state
            # of the model by the concatenating Ewald and short-range embeddings
            # before the linear upprojection layer of each output block
            self.lin_up = nn.Linear(2 * hidden_channels, out_emb_channels, bias=True)
        else:
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

    def forward(self, x, rbf, i, num_nodes=None, h=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        # Combine Ewald part of the model with output
        if self.use_ewald:
            x = torch.cat([x, h], dim=-1)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


@registry.register_model("dpp")
class DimeNetPlusPlus(BaseModel):
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
        force_decoder_model_config (dict): config of the force decoder model.
            keys: "model_type", "hidden_channels", "num_layers", "num_heads",
        force_decoder_type (str): type of the force decoder model.
            (options: "mlp", "simple", "res", "res_updown")
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained"

    def __init__(self, **kwargs):
        super(DimeNetPlusPlus, self).__init__()

        self.cutoff = kwargs["cutoff"]
        self.use_pbc = kwargs["use_pbc"]
        self.otf_graph = kwargs["otf_graph"]
        self.regress_forces = kwargs["regress_forces"]
        self.energy_head = kwargs["energy_head"]
        use_tag = kwargs["tag_hidden_channels"] > 0
        use_pg = kwargs["pg_hidden_channels"] > 0
        self.use_ewald = kwargs.get("use_ewald", False)
        self.detach_ewald = kwargs.get("detach_ewald", True)
        self.use_atom_to_atom_mp = kwargs.get("use_atom_to_atom_mp", False)
        act = (
            getattr(nn.functional, kwargs["act"]) if kwargs["act"] != "swish" else swish
        )

        assert (
            kwargs["tag_hidden_channels"] + 2 * kwargs["pg_hidden_channels"] + 16
            < kwargs["hidden_channels"]
        )
        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        self.rbf = BesselBasisLayer(
            kwargs["num_radial"], self.cutoff, kwargs["envelope_exponent"]
        )
        self.sbf = SphericalBasisLayer(
            kwargs["num_spherical"],
            kwargs["num_radial"],
            self.cutoff,
            kwargs["envelope_exponent"],
        )

        if use_tag or use_pg or kwargs["phys_embeds"] or kwargs["graph_rewiring"]:
            self.emb = AdvancedEmbeddingBlock(
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
            self.emb = EmbeddingBlock(
                kwargs["num_radial"], kwargs["hidden_channels"], act
            )

        if self.energy_head:
            self.output_blocks = torch.nn.ModuleList(
                [
                    EHOutputPPBlock(
                        kwargs["num_radial"],
                        kwargs["hidden_channels"],
                        kwargs["out_emb_channels"],
                        kwargs["num_targets"],
                        kwargs["num_output_layers"],
                        self.energy_head,
                        act,
                        self.use_ewald,
                    )
                    for _ in range(kwargs["num_blocks"] + 1)
                ]
            )
        else:
            self.output_blocks = torch.nn.ModuleList(
                [
                    OutputPPBlock(
                        kwargs["num_radial"],
                        kwargs["hidden_channels"],
                        kwargs["out_emb_channels"],
                        kwargs["num_targets"],
                        kwargs["num_output_layers"],
                        act,
                        self.use_ewald,
                    )
                    for _ in range(kwargs["num_blocks"] + 1)
                ]
            )

        self.interaction_blocks = torch.nn.ModuleList(
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

        if kwargs.get("use_ewald", False):
            # Initialize k-space structure
            if self.use_pbc:
                # Get the reciprocal lattice indices of included k-vectors
                (
                    self.k_index_product_set,
                    self.num_k_degrees_of_freedom,
                ) = get_k_index_product_set(
                    kwargs["ewald_hyperparams"]["num_k_x"],
                    kwargs["ewald_hyperparams"]["num_k_y"],
                    kwargs["ewald_hyperparams"]["num_k_z"],
                )
                self.k_rbf_values = None
                self.delta_k = None

            else:
                # Get the k-space voxel and evaluate Gaussian RBF (can be done at
                # initialization time as voxel grid stays fixed for all structures)
                (
                    self.k_grid,
                    self.k_rbf_values,
                    self.num_k_degrees_of_freedom,
                ) = get_k_voxel_grid(
                    kwargs["ewald_hyperparams"]["k_cutoff"],
                    self.delta_k,
                    kwargs["ewald_hyperparams"]["num_k_rbf"],
                )

            # Initialize atom embedding block
            self.atom_emb = AtomEmbedding(kwargs["hidden_channels"], num_elements=83)

            # Downprojection layer, weights are shared among all interaction blocks
            self.down = Dense(
                self.num_k_degrees_of_freedom,
                kwargs["ewald_hyperparams"]["downprojection_size"],
                activation=None,
                bias=False,
            )

            self.ewald_blocks = torch.nn.ModuleList(
                [
                    EwaldBlock(
                        self.down,
                        kwargs["hidden_channels"],  # Embedding size of short-range GNN
                        kwargs["ewald_hyperparams"]["downprojection_size"],
                        kwargs["ewald_hyperparams"][
                            "num_hidden"
                        ],  # Number of residuals in update function
                        activation="silu",
                        use_pbc=self.use_pbc,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(kwargs["num_blocks"])
                ]
            )

        if self.use_atom_to_atom_mp:
            self.atom_emb = AtomEmbedding(kwargs["hidden_channels"], num_elements=83)
            if self.use_pbc:
                # Compute neighbor threshold from cutoff assuming uniform atom density
                self.max_neighbors_at = int((self.cutoff / 6.0) ** 3 * 50)
            else:
                self.max_neighbors_at = 100
            # SchNet interactions for atom-to-atom message passing
            self.interactions_at = torch.nn.ModuleList(
                [
                    InteractionBlock(
                        kwargs["hidden_channels"],
                        200,  # num Gaussians
                        256,  # num filters
                        self.cutoff,
                    )
                    for i in range(kwargs["num_blocks"])
                ]
            )
            self.distance_expansion_at = GaussianSmearing(0.0, self.cutoff, 200)

        self.skip_connection_factor = (
            1.0 + float(self.use_ewald) + float(self.use_atom_to_atom_mp)
        ) ** (-0.5)

        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(kwargs["hidden_channels"], 1)

        self.task = kwargs["task_name"]

        # Force head
        self.decoder = (
            ForceDecoder(
                kwargs["force_decoder_type"],
                kwargs["hidden_channels"],
                kwargs["force_decoder_model_config"],
                act,
            )
            if "direct" in self.regress_forces
            else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin.bias.data.fill_(0)
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
    def energy_forward(self, data, q=None):
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        # Rewire the graph
        pos = data.pos
        batch = data.batch
        batch_size = int(batch.max()) + 1
        if not hasattr(data, "subnodes"):
            data.subnodes = False

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_offsets=True,
            )

            edge_index = out["edge_index"]
            dist = out["distances"]
            offsets = out["offsets"]

            j, i = edge_index
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            j, i = edge_index
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        if (
            self.task == "qm9" and edge_index.shape[1] != len(data.cell_offsets)
        ) or self.task == "qm7x":
            data.cell_offsets = torch.zeros(
                (edge_index.shape[1], 3), device=edge_index.device
            )

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index,
            data.cell_offsets,
            num_nodes=data.atomic_numbers.size(0),
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        if self.use_pbc:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i + offsets[idx_ji],
                pos[idx_k].detach() - pos_j + offsets[idx_kj],
            )
        else:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i,
                pos[idx_k].detach() - pos_j,
            )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        if self.use_ewald:
            if self.use_pbc:
                # Compute reciprocal lattice basis of structure
                k_cell, _ = x_to_k_cell(data.cell)
                # Translate lattice indices to k-vectors
                k_grid = torch.matmul(self.k_index_product_set.to(batch.device), k_cell)
            else:
                k_grid = (
                    self.k_grid.to(batch.device).unsqueeze(0).expand(batch_size, -1, -1)
                )

        if self.use_atom_to_atom_mp:
            # Use separate graph (larger cutoff) for atom-to-atom long-range block
            (
                edge_index_at,
                edge_weight_at,
                distance_vec_at,
                cell_offsets_at,
                _,  # cell offset distances
                neighbors_at,
            ) = self.generate_graph(
                data,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors_at,
            )

            edge_attr_at = self.distance_expansion_at(edge_weight_at)

        # Embedding block.
        x = self.emb(data.atomic_numbers.long(), rbf, i, j, data.tags, data.subnodes)

        if self.use_ewald:
            # If Ewald MP is used, we have to create atom embeddings borrowing
            # the atomic embedding block from the GemNet architecture
            h = self.atom_emb(data.atomic_numbers.long())
            dot = None  # These will be computed in first Ewald block and then passed
            sinc_damping = (
                None  # on between later Ewald blocks (avoids redundant recomputation)
            )
            pos_detach = pos.detach() if self.detach_ewald else pos
        elif self.use_atom_to_atom_mp:
            h = self.atom_emb(data.atomic_numbers.long())
        else:
            h = None

        if self.energy_head:
            P, batch = self.output_blocks[0](
                x, rbf, i, edge_index, dist, data.batch, num_nodes=pos.size(0), h=h
            )
        else:
            P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0), h=h)

        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(scatter(x, i, dim=0, dim_size=pos.size(0)))

        # Interaction blocks.

        energy_Ps = []

        if self.use_ewald or self.use_atom_to_atom_mp:
            for block_ind in range(len(self.interaction_blocks)):
                x = self.interaction_blocks[block_ind](x, rbf, sbf, idx_kj, idx_ji)

                if self.use_ewald:
                    h_ewald, dot, sinc_damping = self.ewald_blocks[block_ind](
                        h,
                        pos_detach,
                        k_grid,
                        batch_size,
                        batch,
                        dot,
                        sinc_damping,
                    )
                else:
                    h_ewald = 0

                if self.use_atom_to_atom_mp:
                    h_at = self.interactions_at[block_ind](
                        h, edge_index_at, edge_weight_at, edge_attr_at
                    )
                else:
                    h_at = 0

                h = self.skip_connection_factor * (h + h_ewald + h_at)
                if self.energy_head:
                    P_bis, _ = self.output_blocks[block_ind + 1](
                        x, rbf, i, num_nodes=pos.size(0), h=h
                    )
                    energy_Ps.append(
                        P_bis.sum(0) / len(P)
                        if batch is None
                        else scatter(P_bis, batch, dim=0)
                    )
                else:
                    P += self.output_blocks[block_ind + 1](
                        x, rbf, i, num_nodes=pos.size(0), h=h
                    )

        else:
            for block_ind, (interaction_block, output_block) in enumerate(
                zip(self.interaction_blocks, self.output_blocks[1:])
            ):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)

                if self.energy_head:
                    P_bis, _ = output_block(
                        x, rbf, i, edge_index, dist, data.batch, num_nodes=pos.size(0)
                    )
                    energy_Ps.append(
                        P_bis.sum(0) / len(P)
                        if batch is None
                        else scatter(P_bis, batch, dim=0)
                    )
                else:
                    P += output_block(x, rbf, i, num_nodes=pos.size(0))

        P_bis = sum(energy_Ps or [0])

        if self.energy_head == "weighted-av-initial-embeds":
            P = P * alpha

        # Output
        # scatter
        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        energy = energy + P_bis

        return {
            "energy": energy,
            "hidden_state": scatter(x, i, dim=0, dim_size=pos.size(0)),
        }

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        return self.decoder(preds["hidden_state"])

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
