import torch
import math
from torch import nn
from torch.nn import Linear, Transformer, Softmax

from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph, GATConv, GATv2Conv

from torch_sparse import SparseTensor, spspmm
from torch_sparse import transpose as transpose_sparse
from scipy import sparse

from ocpmodels.models.faenet import (
    GaussianSmearing,
    EmbeddingBlock,
    InteractionBlock,
    OutputBlock
)
from ocpmodels.models.indfaenet import PositionalEncoding
from ocpmodels.common.registry import registry
from ocpmodels.models.base_model import BaseModel
from ocpmodels.common.utils import conditional_grad, get_pbc_distances
from ocpmodels.models.utils.activations import swish

class GATInteraction(nn.Module):
    def __init__(self, d_model, version, edge_dim, dropout=0.1):
        super(GATInteraction, self).__init__()

        if version not in {"v1", "v2"}:
            raise ValueError(f"Invalid GAT version. Received {version}, available: v1, v2.")

        # Not quite sure what is the impact of increasing or decreasing the number of heads
        if version == "v1":
            self.interaction = GATConv(
                in_channels = d_model,
                out_channels = d_model,
                heads = 3,
                concat = False,
                edge_dim = edge_dim,
                dropout = dropout
            )
        else:
            self.interaction = GATv2Conv(
                in_channels = d_model,
                out_channels = d_model,
                head = 3,
                concat = False,
                edge_dim = edge_dim,
                dropout = dropout
            )

    def forward(self, h_ads, h_cat, bipartite_edges, bipartite_weights):
        # We first do the message passing
        separation_pt = h_ads.shape[0]
        combined = torch.concat([h_ads, h_cat], dim = 0)
        combined = self.interaction(combined, bipartite_edges, bipartite_weights)

        # Then we normalize and add residual connections
        ads, cat = combined[:separation_pt], combined[separation_pt:]
        ads, cat = nn.functional.normalize(ads), nn.functional.normalize(cat)
        ads, cat = ads + h_ads, cat + h_cat
        # QUESTION: Should normalization happen before separating them?

        return ads, cat

@registry.register_model("afaenet")
class AFaenet(BaseModel):
    def __init__(self, **kwargs):
        super(AFaenet, self).__init__()

        self.cutoff = kwargs["cutoff"]
        self.energy_head = kwargs["energy_head"]
        self.regress_forces = kwargs["regress_forces"]
        self.use_pbc = kwargs["use_pbc"]
        self.max_num_neighbors = kwargs["max_num_neighbors"]
        self.edge_embed_type = kwargs["edge_embed_type"]
        self.skip_co = kwargs["skip_co"]
        if kwargs["mp_type"] == "sfarinet":
            kwargs["num_filters"] = kwargs["hidden_channels"]
        self.hidden_channels = kwargs["hidden_channels"]

        self.act = (
            getattr(nn.functional, kwargs["act"]) if kwargs["act"] != "swish" else swish
        )
        self.use_positional_embeds = kwargs["graph_rewiring"] in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }

        # Gaussian Basis
        self.distance_expansion_ads = GaussianSmearing(
            0.0, self.cutoff, kwargs["num_gaussians"]
        )
        self.distance_expansion_cat = GaussianSmearing(
            0.0, self.cutoff, kwargs["num_gaussians"]
        )
        self.distance_expansion_disc = GaussianSmearing(
            0.0, 20.0, kwargs["num_gaussians"] 
        )
        # Set the second parameter as the highest possible z-axis value

        # Embedding block
        self.embed_block_ads = EmbeddingBlock(
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
        self.embed_block_cat = EmbeddingBlock(
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
        self.disc_edge_embed = Linear(kwargs["num_gaussians"], kwargs["num_filters"] // 2)

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

        assert "afaenet_gat_mode" in kwargs, "GAT version needs to be specified. Options: v1, v2"
        # Inter Interaction
        self.inter_interactions = nn.ModuleList(
            [
                GATInteraction(
                    kwargs["hidden_channels"],
                    kwargs["afaenet_gat_mode"],
                    kwargs["num_filters"] // 2,
                )
                for _ in range(kwargs["num_interactions"])
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

        self.transformer_out = kwargs.get("transformer_out", False)
        if self.transformer_out:
            self.combination = Transformer(
                d_model = kwargs["hidden_channels"] // 2,
                nhead = 2,
                num_encoder_layers = 2,
                num_decoder_layers = 2,
                dim_feedforward = kwargs["hidden_channels"],
                batch_first = True
            )
            self.positional_encoding = PositionalEncoding(
                kwargs["hidden_channels"] // 2,
                dropout = 0.1,
                max_len = 5,
            )
            self.query_pos = nn.Parameter(torch.rand(kwargs["hidden_channels"] // 2))
            self.transformer_lin = Linear(kwargs["hidden_channels"] // 2, 1)
        else:
            self.combination = nn.Sequential(
                Linear(kwargs["hidden_channels"], kwargs["hidden_channels"] // 2),
                self.act,
                Linear(kwargs["hidden_channels"] // 2, 1)
            )

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        batch_size = len(data)
        batch_ads = data["adsorbate"]["batch"]
        batch_cat = data["catalyst"]["batch"]

        # Graph rewiring
        ads_rewiring, cat_rewiring = self.graph_rewiring(data, batch_ads, batch_cat)
        edge_index_ads, edge_weight_ads, rel_pos_ads, edge_attr_ads = ads_rewiring
        edge_index_cat, edge_weight_cat, rel_pos_cat, edge_attr_cat = cat_rewiring

        # Embedding
        h_ads, e_ads = self.embedding(
            data["adsorbate"].atomic_numbers.long(),
            edge_weight_ads, 
            rel_pos_ads,
            edge_attr_ads,
            data["adsorbate"].tags,
            self.embed_block_ads
        )
        h_cat, e_cat = self.embedding(
            data["catalyst"].atomic_numbers.long(),
            edge_weight_cat,
            rel_pos_cat,
            edge_attr_cat,
            data["catalyst"].tags,
            self.embed_block_cat
        )

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha_ads = self.w_lin_ads(h_ads)
            alpha_cat = self.w_lin_cat(h_cat)
        else:
            alpha_ads = None
            alpha_cat = None

        # Edge embeddings of the complete bipartite graph.
        edge_weights = self.distance_expansion_disc(data["is_disc"].edge_weight)
        edge_weights = self.disc_edge_embed(edge_weights)

        # Now we do interactions.
        energy_skip_co_ads = []
        energy_skip_co_cat = []
        for (
            interaction_ads,
            interaction_cat,
            inter_interaction
        ) in zip(
            self.interaction_blocks_ads,
            self.interaction_blocks_cat,
            self.inter_interactions,
        ):
            if self.skip_co == "concat_atom":
                energy_skip_co_ads.append(h_ads)
                energy_skip_co_cat.append(h_cat)
            elif self.skip_co:
                energy_skip_co_ads.append(
                    self.output_block_ads(
                        h_ads, edge_index_ads, edge_weight_ads, batch_ads, alpha_ads
                    )
                )
                energy_skip_co_cat.append(
                    self.output_block_cat(
                        h_cat, edge_index_cat, edge_weight_cat, batch_cat, alpha_cat
                    )
                )
            # First we do intra interaction
            intra_ads = interaction_ads(h_ads, edge_index_ads, e_ads)
            intra_cat = interaction_cat(h_cat, edge_index_cat, e_cat)

            # Then we do inter interaction
            h_ads, h_cat = inter_interaction(
                intra_ads,
                intra_cat,
                data["is_disc"].edge_index,
                edge_weights,
            )
            # QUESTION: Can we do both simultaneously?

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co_ads.append(h_ads)
            energy_skip_co_cat.append(h_cat)

            h_ads = self.act(self.mlp_skip_co_ads(torch.cat(energy_skip_co_ads, dim = 1)))
            h_cat = self.act(self.mlp_skip_co_cat(torch.cat(energy_skip_co_cat, dim = 1)))        

        energy_ads = self.output_block_ads(
            h_ads, edge_index_ads, edge_weight_ads, batch_ads, alpha_ads
        )
        energy_cat = self.output_block_cat(
            h_cat, edge_index_cat, edge_weight_cat, batch_cat, alpha_cat
        )

        # Skip-connection
        energy_skip_co_ads.append(energy_ads)
        energy_skip_co_cat.append(energy_cat)
        if self.skip_co == "concat":
            energy_ads = self.mlp_skip_co_ads(torch.cat(energy_skip_co_ads, dim = 1))
            energy_cat = self.mlp_skip_co_cat(torch.cat(energy_skip_co_cat, dim = 1))
        elif self.skip_co == "add":
            energy_ads = sum(energy_skip_co_ads)
            energy_cat = sum(energy_skip_co_cat)

        # Combining hidden representations
        if self.transformer_out:
            batch_size = energy_ads.shape[0]
            
            fake_target_sequence = self.query_pos.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
            system_energy = torch.cat(
                [
                    energy_ads.unsqueeze(1),
                    energy_cat.unsqueeze(1)
                ],
                dim = 1
            )

            system_energy = self.positional_encoding(system_energy)
            
            system_energy = self.combination(system_energy, fake_target_sequence).squeeze(1)
            system_energy = self.transformer_lin(system_energy)
        else:
            system_energy = torch.cat([energy_ads, energy_cat], dim = 1)
            system_energy = self.combination(system_energy)

        # We combine predictions and return them
        pred_system = {
            "energy" : system_energy,
            "pooling_loss" : None, # This might break something.
            "hidden_state" : torch.cat([energy_ads, energy_cat], dim = 1)
        }

        return pred_system

    @conditional_grad(torch.enable_grad())
    def embedding(self, z, edge_weight, rel_pos, edge_attr, tags, embed_func):
        # Normalize and squash to [0,1] for gaussian basis
        rel_pos_normalized = None
        if self.edge_embed_type in {"sh", "all_rij", "all"}:
            rel_pos_normalized = (rel_pos / edge_weight.view(-1, 1) + 1) / 2.0

        pooling_loss = None  # deal with pooling loss

        # Embedding block
        h, e = embed_func(z, rel_pos, edge_attr, tags, rel_pos_normalized)

        return h, e

    @conditional_grad(torch.enable_grad())
    def graph_rewiring(self, data, batch_ads, batch_cat):
        z = data["adsorbate"].atomic_numbers.long()

        # Use periodic boundary conditions
        results = []
        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            for mode in ["adsorbate", "catalyst"]:
                out = get_pbc_distances(
                    data[mode].pos,
                    data[mode, "is_close", mode].edge_index,
                    data[mode].cell,
                    data[mode].cell_offsets,
                    data[mode].neighbors,
                    return_distance_vec=True,
                )

                edge_index = out["edge_index"]
                edge_weight = out["distances"]
                rel_pos = out["distance_vec"]
                if mode == "adsorbate":
                    distance_expansion = self.distance_expansion_ads
                else:
                    distance_expansion = self.distance_expansion_cat 
                edge_attr = distance_expansion(edge_weight)
                results.append([edge_index, edge_weight, rel_pos, edge_attr])
        else:
            for mode in ["adsorbate", "catalyst"]:
                edge_index = radius_graph(
                    data[mode].pos,
                    r=self.cutoff,
                    batch=batch_ads if mode == "adsorbate" else batch_cat,
                    max_num_neighbors=self.max_num_neighbors,
                )
                # edge_index = data.edge_index
                row, col = edge_index
                rel_pos = data[mode].pos[row] - data[mode].pos[col]
                edge_weight = rel_pos.norm(dim=-1)
                if mode == "adsorbate":
                    distance_expansion = self.distance_expansion_ads
                else:
                    distance_expansion = self.distance_expansion_cat
                edge_attr = distance_expansion(edge_weight)
                results.append([edge_index, edge_weight, rel_pos, edge_attr])

        return results

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        pass
