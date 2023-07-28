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
                # self.mlp_skip_co_ads = Linear(
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
        

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        pass
