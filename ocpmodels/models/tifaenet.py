import torch
from torch import nn

from ocpmodels.models.faenet import (
    GaussianSmearing,
    EmbeddingBlock,
    InteractionBlock,
    OutputBlock
)
from ocpmodels.common.registry import registry
from ocpmodels.models.base_model import BaseModel

class TransformerInteraction(nn.Module):
    def __init__(self, placeholder):
        pass

    def forward(self, inputs):
        pass

@registry.register_model("tifaenet")
class TIFaenet(BaseModel)
    def __init__(self, **kwargs):
        super(TIFaenet, self).__init__()

        self.cutoff = kwargs["cutoff"]
        self.energy_head = kwargs["energy_head"]
        self.regress_forces = kwargs["regress_forces"]
        self.use_pbc = kwargs["use_pbc"]
        self.max_num_neighbors = kwargs["max_num_neighbors"]
        self.edge_embed_type = kwargs["edge_embed_type"]
        self.skip_co = kwargs["skip_co"]
        if kwargs["mp_type"] == "sfarinet":
            kwargs["num_filters"] = kwargs["hidden_channels"]

        self.act = (
            getattr(nn.functional, kwargs["act"]) if kwargs["act"] != "swish" else swish
        )
        self.use_positional_embeds = kwargs["graph_rewiring"] in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(
            0.0, self.cutoff, kwargs["num_gaussians"]
        )

        # Embedding block
        self.embed_block = EmbeddingBlock(
            kwargs["num_gaussians"],
            kwargs["num_filters"],
            kwargs["hidden_channels"],
            kwargs["tag_hidden_channels"],
            kwargs["pg_hidden_channels"],
            kwargs["phys_hidden_channels"],
            kwargs["phys_embeds"],
            kwargs["graph_rewiring"],
            self.act,
            kwargs["second_layer_MLP"],
            kwargs["edge_embed_type"],
        )

        # Interaction block
        self.interaction_blocks_ads = nn.ModuleList(
            [
                InteractionBlock(
                    kwargs["hidden_channels"],
                    kwargs["num_filters"],
                    self.act,
                    kwargs["mp_type"],
                    kwargs["complex_mp"],
                    kwargs["att_heads"],
                    kwargs["graph_norm"],
                )
                for _ in range(kwargs["num_interactions"])
            ]
        )
        self.interaction_blocks_cat = nn.ModuleList(
            [
                InteractionBlock(
                    kwargs["hidden_channels"],
                    kwargs["num_filters"],
                    self.act,
                    kwargs["mp_type"],
                    kwargs["complex_mp"],
                    kwargs["att_heads"],
                    kwargs["graph_norm"],
                )
                for _ in range(kwargs["num_interactions"])
            ]
        )

        # Transformer Interaction
        self.transformer_blocks_ads = nn.ModuleList(
            [
                TransformerInteraction(
                    placeholder = 3.14159265
                )
            ]
        )

        # Output blocks
        self.output_block_ads = OutputBlock(
            self.energy_head, kwargs["hidden_channels"], self.act, kwargs["model_name"]
        )
        self.output_block_cat = OutputBlock(
            self.energy_head, kwargs["hidden_channels"], self.act, kwargs["model_name"]
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin_ads = Linear(kwargs["hidden_channels"], 1)
            self.w_lin_cat = Linear(kwargs["hidden_channels"], 1)

        # Skip co
        if self.skip_co == "concat": # for the implementation of independent faenet, make sure the input is large enough
            if kwargs["model_name"] in {"faenet", "depfaenet"}:
                self.mlp_skip_co_ads = Linear(
                    kwargs["num_interactions"] + 1,
                    1
                )
                self.mlp_skip_co_cat = Linear(
                    kwargs["num_interactions"] + 1,
                    1
                )
            elif kwargs["model_name"] == "indfaenet":
                self.mlp_skip_co_ads = Linear(
                    (kwargs["num_interactions"] + 1) * kwargs["hidden_channels"] // 2,
                    kwargs["hidden_channels"] // 2
                )
                self.mlp_skip_co_cat = Linear(
                    (kwargs["num_interactions"] + 1) * kwargs["hidden_channels"] // 2,
                    kwargs["hidden_channels"] // 2
                )

        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = Linear(
                ((kwargs["num_interactions"] + 1) * kwargs["hidden_channels"]),
                kwargs["hidden_channels"],
            )

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        batch_size = len(data) // 2

        adsorbates = Batch.from_data_list(data[:batch_size])
        catalysts = Batch.from_data_list(data[batch_size:])

        # Fixing neighbor's dimensions. This error happens when an adsorbate has 0 edges.
        adsorbates = self.neighbor_fixer(adsorbates)
        catalysts = self.neighbor_fixer(catalysts)

        # Graph rewiring
        ads_rewiring = graph_rewiring(adsorbates)
        edge_index_ads, edge_weight_ads, rel_pos_ads, edge_attr_ads = ads_rewiring

        cat_rewiring = graph_rewiring(catalysts)
        edge_index_cat, edge_weight_cat, rel_pos_cat, edge_attr_cat = cat_rewiring

    @conditional_grad(torch.enable_grad())
    def graph_rewiring(self, data)
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        # Use periodic boundary conditions
        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            rel_pos = out["distance_vec"]
            edge_attr = self.distance_expansion(edge_weight)
        else:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            # edge_index = data.edge_index
            row, col = edge_index
            rel_pos = pos[row] - pos[col]
            edge_weight = rel_pos.norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)

        return (edge_index, edge_weight, rel_pos, edge_attr)

    def neighbor_fixer(self, data):
        num_graphs = len(data)
        # Find indices of adsorbates without edges:
        edgeless = [
            i for i 
            in range(num_graphs) 
            if data[i].neighbors.shape[0] == 0
        ]
        if len(edgeless) > 0:
            # Since most adsorbates have an edge,
            # we pop those values specifically from range(num_adsorbates)
            mask = list(range(num_graphs))
            num_popped = 0 # We can do this since edgeless is already sorted
            for unwanted in edgeless:
                mask.pop(unwanted-num_popped)
                num_popped += 1
            new_nbrs = torch.zeros(
                num_graphs,
                dtype = torch.int64,
                device = data.neighbors.device,
            )
            new_nbrs[mask] = data.neighbors
            data.neighbors = new_nbrs

        return data

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        pass
