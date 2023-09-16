"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch import nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.modules.pooling import Graclus, Hierarchical_Pooling
from ocpmodels.models.utils.activations import swish
from ocpmodels.models.schnet import (
    InteractionBlock,
    CFConv,
    GaussianSmearing,
    ShiftedSoftplus,
)
from ocpmodels.models.afaenet import GATInteraction

NUM_CLUSTERS = 20
NUM_POOLING_LAYERS = 1

@registry.register_model("aschnet")
class ASchNet(BaseModel):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        use_pbc (bool, optional): Use of periodic boundary conditions.
            (default: true)
        otf_graph (bool, optional): Recompute radius graph.
            (default: false)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        graph_rewiring (str, optional): Method used to create the graph,
            among "", remove-tag-0, supernodes.
        energy_head (str, optional): Method to compute energy prediction
            from atom representations.
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        tag_hidden_channels (int, optional): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int, optional): Hidden period and group embed size.
            (default: obj:`32`)
        phys_embed (bool, optional): Concat fixed physics-aware embeddings.
        phys_hidden_channels (int, optional): Hidden size of learnable phys embed.
            (default: obj:`32`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"

    def __init__(self, **kwargs):
        super().__init__()

        import ase

        self.use_pbc = kwargs["use_pbc"]
        self.cutoff = kwargs["cutoff"]
        self.otf_graph = kwargs["otf_graph"]
        self.scale = None
        self.regress_forces = kwargs["regress_forces"]

        self.num_filters = kwargs["num_filters"]
        self.num_interactions = kwargs["num_interactions"]
        self.num_gaussians = kwargs["num_gaussians"]
        self.max_num_neighbors = kwargs["max_num_neighbors"]
        self.readout = kwargs["readout"]
        self.hidden_channels = kwargs["hidden_channels"]
        self.tag_hidden_channels = kwargs["tag_hidden_channels"]
        self.use_tag = self.tag_hidden_channels > 0
        self.pg_hidden_channels = kwargs["pg_hidden_channels"]
        self.use_pg = self.pg_hidden_channels > 0
        self.phys_hidden_channels = kwargs["phys_hidden_channels"]
        self.energy_head = kwargs["energy_head"]
        self.use_phys_embeddings = kwargs["phys_embeds"]
        self.use_mlp_phys = self.phys_hidden_channels > 0 and kwargs["phys_embeds"]
        self.use_positional_embeds = kwargs["graph_rewiring"] in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }

        self.register_buffer(
            "initial_atomref",
            torch.tensor(kwargs["atomref"]) if kwargs["atomref"] is not None else None,
        )
        self.atomref = None
        if kwargs["atomref"] is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(torch.tensor(kwargs["atomref"]))

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        # self.covalent_radii = torch.from_numpy(ase.data.covalent_radii)
        # self.vdw_radii = torch.from_numpy(ase.data.vdw_radii)
        self.register_buffer("atomic_mass", atomic_mass)

        if self.use_tag:
            self.tag_embedding = Embedding(3, self.tag_hidden_channels)

        # Phys embeddings
        self.phys_emb = PhysEmbedding(props=kwargs["phys_embeds"], pg=self.use_pg)
        if self.use_mlp_phys:
            self.phys_lin = Linear(
                self.phys_emb.n_properties, self.phys_hidden_channels
            )
        else:
            self.phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = Embedding(
                self.phys_emb.period_size, self.pg_hidden_channels
            )
            self.group_embedding = Embedding(
                self.phys_emb.group_size, self.pg_hidden_channels
            )

        assert (
            self.tag_hidden_channels
            + 2 * self.pg_hidden_channels
            + self.phys_hidden_channels
            < self.hidden_channels
        )

        # Main embedding
        self.embedding_ads = Embedding(
            85,
            self.hidden_channels
            - self.tag_hidden_channels
            - self.phys_hidden_channels
            - 2 * self.pg_hidden_channels,
        )
        self.embedding_cat = Embedding(
            85,
            self.hidden_channels
            - self.tag_hidden_channels
            - self.phys_hidden_channels
            - 2 * self.pg_hidden_channels,
        )

        # Gaussian basis and linear transformation of disc edges
        self.distance_expansion_disc = GaussianSmearing(
            0.0, 20.0, self.num_gaussians
        )
        self.disc_edge_embed = Linear(self.num_gaussians, self.num_filters)

        # Position encoding
        if self.use_positional_embeds:
            self.pe = PositionalEncoding(self.hidden_channels, 210)

        # Interaction block
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)
        
        self.interactions_ads = ModuleList()
        for _ in range(self.num_interactions):
            block = InteractionBlock(
                self.hidden_channels, self.num_gaussians, self.num_filters, self.cutoff
            )
            self.interactions_ads.append(block)

        self.interactions_cat = ModuleList()
        for _ in range(self.num_interactions):
            block = InteractionBlock(
                self.hidden_channels, self.num_gaussians, self.num_filters, self.cutoff
            )
            self.interactions_cat.append(block)

        self.interactions_disc = ModuleList()
        assert "gat_mode" in kwargs, "GAT version needs to be specified. Options: v1, v2"
        for _ in range(self.num_interactions):
            block = GATInteraction(
                self.hidden_channels, kwargs["gat_mode"], self.num_filters
            )
            self.interactions_disc.append(block)

        # Output block
        self.lin1_ads = Linear(self.hidden_channels, self.hidden_channels // 2)
        self.lin1_cat = Linear(self.hidden_channels, self.hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2_ads = Linear(self.hidden_channels // 2, self.hidden_channels // 2)
        self.lin2_cat = Linear(self.hidden_channels // 2, self.hidden_channels // 2)

        # weighted average & pooling
        if self.energy_head in {"pooling", "random"}:
            self.hierarchical_pooling = Hierarchical_Pooling(
                self.hidden_channels,
                self.act,
                NUM_POOLING_LAYERS,
                NUM_CLUSTERS,
                self.energy_head,
            )
        elif self.energy_head == "graclus":
            self.graclus = Graclus(self.hidden_channels, self.act)
        elif self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            self.w_lin = Linear(self.hidden_channels, 1)

        self.combination = nn.Sequential(
            Linear(self.hidden_channels, self.hidden_channels // 2),
            swish,
            Linear(kwargs["hidden_channels"] // 2, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_ads.reset_parameters()
        self.embedding_cat.reset_parameters()
        if self.use_mlp_phys:
            torch.nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        if self.energy_head in {"weighted-av-init-embeds", "weighted-av-final-embeds"}:
            self.w_lin.bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.w_lin.weight)
        for (
            interaction_ads,
            interaction_cat,
            interaction_disc
        ) in zip (
            self.interactions_ads,
            self.interactions_cat,
            self.interactions_disc
        ):
            interaction_ads.reset_parameters()
            interaction_cat.reset_parameters()
            #interaction_disc.reset_parameters() # need to implement this!
        torch.nn.init.xavier_uniform_(self.lin1_ads.weight)
        self.lin1_ads.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2_ads.weight)
        self.lin2_ads.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin1_cat.weight)
        self.lin1_cat.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2_cat.weight)
        self.lin2_cat.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"tag_hidden_channels={self.tag_hidden_channels}, "
            f"properties={self.phys_hidden_channels}, "
            f"period_hidden_channels={self.pg_hidden_channels}, "
            f"group_hidden_channels={self.pg_hidden_channels}, "
            f"energy_head={self.energy_head}",
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})",
        )

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        return

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        """"""
        # Re-compute on the fly the graph
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc_inputs(
                data["adsorbate"].pos,
                data["adsorbate"].natoms, 
                data["adsorbate"].cell,
                self.cutoff,
                50,
            )
            data["adsorbate", "is_close", "adsorbate"].edge_index = edge_index
            data["adsorbate"].cell_offsets = cell_offsets
            data["adsorbate"].neighbors = neighbors
    
            edge_index, cell_offsets, neighbors = radius_graph_pbc_inputs(
                data["catalyst"].pos,
                data["catalyst"].natoms,
                data["catalyst"].cell,
                self.cutoff,
                50,
            )
            data["catalyst", "is_close", "catalyst"].edge_index = edge_index
            data["catalyst"].cell_offsets = cell_offsets
            data["catalyst"].neighbors = neighbors

        # Rewire the graph
        # Use periodic boundary conditions
        ads_rewiring, cat_rewiring = self.graph_rewiring(data, ) 
        edge_index_ads, edge_weight_ads, edge_attr_ads = ads_rewiring
        edge_index_cat, edge_weight_cat, edge_attr_cat = cat_rewiring

        h_ads = self.embedding_ads(data["adsorbate"].atomic_numbers.long())
        h_cat = self.embedding_cat(data["catalyst"].atomic_numbers.long())

        edge_weights_disc = self.distance_expansion_disc(data["is_disc"].edge_weight)
        edge_weights_disc = self.disc_edge_embed(edge_weights_disc)

        if self.use_tag: # NOT IMPLEMENTED
            assert data["adsorbate"].tags is not None
            h_tag = self.tag_embedding(data.tags)
            h = torch.cat((h, h_tag), dim=1)

        if self.phys_emb.device != data["adsorbate"].batch.device: # NOT IMPLEMENTED
            self.phys_emb = self.phys_emb.to(data["adsorbate"].batch.device)

        if self.use_phys_embeddings: # NOT IMPLEMENTED
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        if self.use_pg: # NOT IMPLEMENTED
            # assert self.phys_emb.period is not None
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        if self.use_positional_embeds: # NOT IMPLEMENTED
            idx_of_non_zero_val = (data.tags == 0).nonzero().T.squeeze(0)
            h_pos = torch.zeros_like(h, device=h.device)
            h_pos[idx_of_non_zero_val, :] = self.pe(data.subnodes).to(
                device=h_pos.device
            )
            h += h_pos

        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)

        for (
            interaction_ads,
            interaction_cat,
            interaction_disc
        ) in zip (
            self.interactions_ads,
            self.interactions_cat,
            self.interactions_disc
        ):
            intra_ads = interaction_ads(h_ads, edge_index_ads, edge_weight_ads, edge_attr_ads)
            intra_cat = interaction_cat(h_cat, edge_index_cat, edge_weight_cat, edge_attr_cat)
            inter_ads, inter_cat = interaction_disc(
                intra_ads,
                intra_cat,
                data["is_disc"].edge_index,
                edge_weights_disc
            )
            h_ads, h_cat = h_ads + inter_ads, h_cat + inter_cat
            h_ads, h_cat = nn.functional.normalize(h_ads), nn.functional.normalize(h_cat)

        pooling_loss = None  # deal with pooling loss

        if self.energy_head == "weighted-av-final-embeds": # NOT IMPLEMENTED
            alpha = self.w_lin(h)

        elif self.energy_head == "graclus":
            h, batch = self.graclus(h, edge_index, edge_weight, batch) # NOT IMPLEMENTED

        if self.energy_head in {"pooling", "random"}: # NOT IMPLEMENTED
            h, batch, pooling_loss = self.hierarchical_pooling(
                h, edge_index, edge_weight, batch
            )

        # MLP
        h_ads = self.lin1_ads(h_ads)
        h_ads = self.act(h_ads)
        h_ads = self.lin2_ads(h_ads)

        h_cat = self.lin1_cat(h_cat)
        h_cat = self.act(h_cat)
        h_cat = self.lin2_cat(h_cat)

        if self.energy_head in { # NOT IMPLEMENTED
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        if self.atomref is not None: # NOT IMPLEMENTED
            h = h + self.atomref(z)

        # Global pooling
        out_ads = self.scattering(h_ads, data["adsorbate"].batch)
        out_cat = self.scattering(h_cat, data["catalyst"].batch)

        if self.scale is not None:
            out = self.scale * out

        system = torch.concat([out_ads, out_cat], dim = 1)
        out = self.combination(system)

        return {
            "energy": out,
            "pooling_loss": pooling_loss,
        }

    @conditional_grad(torch.enable_grad())
    def graph_rewiring(self, data):
        results = []

        if self.use_pbc:
            for mode in ["adsorbate", "catalyst"]:
                out = get_pbc_distances(
                    data[mode].pos,
                    data[mode, "is_close", mode].edge_index,
                    data[mode].cell,
                    data[mode].cell_offsets,
                    data[mode].neighbors,
                    return_distance_vec = True
                )

                edge_index = out["edge_index"]
                edge_weight = out["distances"]
                edge_attr = self.distance_expansion(edge_weight)
                results.append([edge_index, edge_weight, edge_attr])
        else:
            for mode in ["adsorbate", "catalyst"]:
                edge_index = radius_graph(
                    data[mode].pos,
                    r = self.cutoff,
                    batch =data[mode].batch,
                    max_num_neighbors = self.max_num_neighbors,
                )
                row, col = edge_index
                edge_weight = (pos[row] - pos[col]).norm(dim=-1)
                edge_attr = self.distance_expansion(edge_weight)
                results.append([edge_index, edge_weight, edge_attr])

        return results

    @conditional_grad(torch.enable_grad())
    def scattering(self, h, batch):
        return scatter(h, batch, dim=0, reduce=self.readout)
