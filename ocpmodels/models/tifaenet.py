import torch
import math
from torch import nn
from torch.nn import Linear, Transformer, Softmax

from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph

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

class TransformerInteraction(nn.Module):
    def __init__(self, d_model, nhead = 2, num_encoder_layers = 2, num_decoder_layers = 2):
        super(TransformerInteraction, self).__init__()

        self.transformer_ads = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = d_model,
            batch_first = True
        )

        self.transformer_ads = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = d_model,
            batch_first = True
        )

    def forward(self, h_ads, h_cat):
        import ipdb
        ipdb.set_trace()


        return h_ads, h_cat
        

class AttentionInteraction(nn.Module):
    def __init__(self, d_model):
        super(AttentionInteraction, self).__init__()

        self.queries_ads = Linear(d_model, d_model)
        self.keys_ads = Linear(d_model, d_model)
        self.values_ads = Linear(d_model, d_model)

        self.queries_cat = Linear(d_model, d_model)
        self.keys_cat = Linear(d_model, d_model)
        self.values_cat = Linear(d_model, d_model)

        self.softmax = Softmax(dim = 1)

    def forward(self, adsorbates, catalysts):
        d_model = adsorbates.h.shape[1]
        batch_size = max(adsorbates.batch).item() + 1

        h_ads = adsorbates.h
        adsorbates.query = self.queries_ads(h_ads)
        adsorbates.key = self.keys_ads(h_ads)
        adsorbates.value = self.values_ads(h_ads)

        h_cat = catalysts.h
        catalysts.query = self.queries_cat(h_cat)
        catalysts.key = self.keys_cat(h_cat)
        catalysts.value = self.values_cat(h_cat)

        new_h_ads = []
        new_h_cat = []
        for i in range(batch_size): # How can I avoid a for loop?
            scalars_ads = self.softmax(
                torch.matmul(adsorbates[i].query, catalysts[i].key.T) / math.sqrt(d_model)
            )
            scalars_cat = self.softmax(
                torch.matmul(catalysts[i].query, adsorbates[i].key.T) / math.sqrt(d_model)
            )

            new_h_ads.append(torch.matmul(scalars_ads, catalysts[i].value))
            new_h_cat.append(torch.matmul(scalars_cat, adsorbates[i].value))

        _, idx = adsorbates.batch.sort(stable=True)
        new_h_ads = torch.concat(new_h_ads, dim = 0)[torch.argsort(idx)] # Inverse of permutation

        _, idx = catalysts.batch.sort(stable=True)
        new_h_cat = torch.concat(new_h_cat, dim = 0)[torch.argsort(idx)]

        new_h_ads = h_ads + new_h_ads
        new_h_cat = h_cat + new_h_cat

        new_h_ads = nn.functional.normalize(new_h_ads)
        new_h_cat = nn.functional.normalize(new_h_cat)

        return new_h_ads, new_h_cat
        

@registry.register_model("tifaenet")
class TIFaenet(BaseModel):
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
        self.distance_expansion_ads = GaussianSmearing(
            0.0, self.cutoff, kwargs["num_gaussians"]
        )
        self.distance_expansion_cat = GaussianSmearing(
            0.0, self.cutoff, kwargs["num_gaussians"]
        )

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
        inter_interaction_type = kwargs.get("tifaenet_mode", None)
        assert inter_interaction_type is not None, "When using TIFaenet, tifaenet_mode is needed. Options: attention, transformer"
        assert inter_interaction_type in {"attention", "transformer"}, "Using an invalid tifaenet_mode. Options: attention, transformer"
        if inter_interaction_type == "transformer":
            inter_interaction_type = TransformerInteraction
            
        elif inter_interaction_type == "attention":
            inter_interaction_type = AttentionInteraction

        self.inter_interactions = nn.ModuleList(
            [
                inter_interaction_type(
                    d_model = kwargs["hidden_channels"],
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
            if kwargs["model_name"] in {"faenet", "depfaenet"}:
                self.mlp_skip_co_ads = Linear(
                    kwargs["num_interactions"] + 1,
                    1
                )
                self.mlp_skip_co_cat = Linear(
                    kwargs["num_interactions"] + 1,
                    1
                )
            elif kwargs["model_name"] in {"indfaenet", "tifaenet"}:
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
        batch_size = len(data) // 2

        adsorbates = Batch.from_data_list(data[:batch_size])
        catalysts = Batch.from_data_list(data[batch_size:])

        batch_ads = adsorbates.batch
        batch_cat = catalysts.batch

        # Fixing neighbor's dimensions. This error happens when an adsorbate has 0 edges.
        adsorbates = self.neighbor_fixer(adsorbates)
        catalysts = self.neighbor_fixer(catalysts)

        # Graph rewiring
        ads_rewiring = self.graph_rewiring(adsorbates)
        edge_index_ads, edge_weight_ads, rel_pos_ads, edge_attr_ads = ads_rewiring

        cat_rewiring = self.graph_rewiring(catalysts)
        edge_index_cat, edge_weight_cat, rel_pos_cat, edge_attr_cat = cat_rewiring

        # Embedding
        h_ads, e_ads = self.embedding(
            adsorbates.atomic_numbers.long(),
            edge_weight_ads, 
            rel_pos_ads,
            edge_attr_ads,
            adsorbates.tags,
            self.embed_block_ads
        )
        h_cat, e_cat = self.embedding(
            catalysts.atomic_numbers.long(),
            edge_weight_cat,
            rel_pos_cat,
            edge_attr_cat,
            catalysts.tags,
            self.embed_block_cat
        )

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha_ads = self.w_lin_ads(h_ads)
            alpha_cat = self.w_lin_cat(h_cat)
        else:
            alpha_ads = None
            alpha_cat = None

        # Interaction and transformer blocks
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
            intra_ads = interaction_ads(h_ads, edge_index_ads, e_ads)
            intra_cat = interaction_cat(h_cat, edge_index_cat, e_cat)

            adsorbates.h = intra_ads
            catalysts.h = intra_cat

            h_ads, h_cat = inter_interaction(adsorbates, catalysts)

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
    def graph_rewiring(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        mode = data.mode[0]
        if mode == "adsorbate":
            distance_expansion = self.distance_expansion_ads
        else:
            distance_expansion = self.distance_expansion_cat
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
            edge_attr = distance_expansion(edge_weight)
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
            edge_attr = distance_expansion(edge_weight)

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
