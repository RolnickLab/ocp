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
    def __init__(self, d_model, version, dropout=0.1):
        super(GATInteraction, self).__init__()

        if version not in {"v1", "v2"}:
            raise ValueError(f"Invalid GAT version. Received {version}, available: v1, v2.")

        if version == "v1":
            self.interaction = GATConv(
                in_channels = d_model,
                out_channels = d_model,
                heads = 3,
                dropout = dropout
            )
        else:
            self.interaction = GATv2Conv(
                in_channels = d_model,
                out_channels = d_model,
                head = 3,
                dropout = dropout
            )
    def forward(self, h_ads, h_cat, bipartite_edges):
        separation_pt = h_ads.shape[0]
        combined = torch.concat([h_ads, h_cat], dim = 0)
        combined = self.interaction(combined, bipartite_edges)

        ads, cat = combined[:separation_pt], combined[separation_pt:]
        ads, cat = nn.functional.normalize(ads), nn.functional.normalize(cat)
        ads, cat = ads + h_ads, cat + h_cat

        return ads, cat

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

    def forward(self, h_ads, h_cat, ads_to_cat, cat_to_ads):
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

    def forward(self, 
        h_ads, h_cat,
        index_ads, index_cat,
        batch_size
    ):
        d_model = h_ads.shape[1]
        natoms_ads = h_ads.shape[0]
        natoms_cat = h_cat.shape[0]

        # Create matrices with values
        query_ads = self.queries_ads(h_ads)
        key_ads = self.keys_ads(h_ads)
        value_ads = self.values_ads(h_ads)

        query_cat = self.queries_cat(h_cat)
        key_cat = self.keys_cat(h_cat)
        value_cat = self.values_cat(h_cat)

        key_cat_T_index, key_cat_T_val = transpose_sparse(
            index_cat, key_cat.view(-1),
            natoms_cat, d_model * batch_size
        )
        key_ads_T_index, key_ads_T_val = transpose_sparse(
            index_ads, key_ads.view(-1),
            natoms_ads, d_model * batch_size
        )

        index_att_ads, attention_ads = spspmm(
            index_ads, query_ads.view(-1),
            key_cat_T_index, key_cat_T_val,
            natoms_ads, d_model * batch_size, natoms_cat
        )
        attention_ads = SparseTensor(
            row=index_att_ads[0], col=index_att_ads[1], value=attention_ads
        ).to_dense()
        attention_ads = self.softmax(attention_ads / math.sqrt(d_model))
        new_h_ads = torch.matmul(attention_ads, value_cat)

        index_att_cat, attention_cat = spspmm(
            index_cat, query_cat.view(-1),
            key_ads_T_index, key_ads_T_val,
            natoms_cat, d_model * batch_size, natoms_ads
        )
        attention_cat = SparseTensor(
            row=index_att_cat[0], col=index_att_cat[1], value=attention_cat
        ).to_dense()
        attention_cat = self.softmax(attention_cat / math.sqrt(d_model))
        new_h_cat = torch.matmul(attention_cat, value_ads)

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
        self.inter_interaction_type = inter_interaction_type
        assert inter_interaction_type is not None, "When using TIFaenet, tifaenet_mode is needed. Options: attention, transformer"
        assert inter_interaction_type in {"attention", "transformer", "gat"}, "Using an invalid tifaenet_mode. Options: attention, transformer, gat"
        if inter_interaction_type == "transformer":
            inter_interaction_type = TransformerInteraction
            
        elif inter_interaction_type == "attention":
            inter_interaction_type = AttentionInteraction
            inter_interaction_parameters = [kwargs["hidden_channels"]]

        elif inter_interaction_type == "gat":
            assert "tifaenet_gat_mode" in kwargs, "When using GAT mode, a version needs to be specified. Options: v1, v2."
            inter_interaction_type = GATInteraction
            inter_interaction_parameters = [
                kwargs["hidden_channels"],
                kwargs["tifaenet_gat_mode"]
            ]

        self.inter_interactions = nn.ModuleList(
            [
                inter_interaction_type(*inter_interaction_parameters)
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

        # Interaction and transformer blocks

        if self.inter_interaction_type == "attention":    
            # Start by setting up the sparse matrices in scipy
            natoms_ads = h_ads.shape[0]
            natoms_cat = h_cat.shape[0]

            dummy_ads = torch.arange(natoms_ads * self.hidden_channels).numpy()
            dummy_cat = torch.ones(natoms_cat * self.hidden_channels).numpy()

            crowd_indices_ads = torch.arange(
                start = 0, end = (natoms_ads + 1)*self.hidden_channels, step = self.hidden_channels,
            ).numpy()
            crowd_indices_cat = torch.arange(
                start = 0, end = (natoms_cat + 1)*self.hidden_channels, step = self.hidden_channels,
            ).numpy()

            raw_col_indices = [
                [torch.arange(self.hidden_channels) + (10*j)] * i
                for i, j
                in zip(adsorbates.natoms, range(batch_size))
            ]
            col_indices = []
            for graph in raw_col_indices:
                col_indices += graph
            col_indices_ads = torch.concat(col_indices).numpy()

            raw_col_indices = [
                [torch.arange(self.hidden_channels) + (10*j)] * i
                for i, j
                in zip(catalysts.natoms, range(batch_size))
            ]
            col_indices = []
            for graph in raw_col_indices:
                col_indices += graph
            col_indices_cat = torch.concat(col_indices).numpy()

            sparse_ads = sparse.csr_array(
                (dummy_ads, col_indices_ads, crowd_indices_ads), shape=(natoms_ads, dummy_ads.shape[0])
            ).tocoo()
            row_ads, col_ads = torch.from_numpy(sparse_ads.row), torch.from_numpy(sparse_ads.col)
            index_ads = torch.concat([row_ads.view(1, -1), col_ads.view(1, -1)], dim=0).long().to(h_ads.device)

            sparse_cat = sparse.csr_array(
                (dummy_cat, col_indices_cat, crowd_indices_cat), shape=(natoms_cat, dummy_cat.shape[0])
            ).tocoo()
            row_cat, col_cat = torch.from_numpy(sparse_cat.row), torch.from_numpy(sparse_cat.col)
            index_cat = torch.concat([row_cat.view(1, -1), col_cat.view(1, -1)], dim=0).long().to(h_ads.device)

            extra_parameters = [index_ads, index_cat, batch_size]
        elif self.inter_interaction_type == "gat":
            extra_parameters = [data["is_disc"].edge_index]
            # Fix edges between graphs

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
            intra_ads = interaction_ads(h_ads, edge_index_ads, e_ads)
            intra_cat = interaction_cat(h_cat, edge_index_cat, e_cat)

            h_ads, h_cat = inter_interaction(
                intra_ads, intra_cat, 
                *extra_parameters
            )

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
