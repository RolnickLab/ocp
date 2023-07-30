import torch, math
from torch import nn
from torch.nn import Linear, Transformer

from ocpmodels.models.faenet import FAENet
from ocpmodels.models.faenet import OutputBlock
from ocpmodels.models.base_model import BaseModel
from ocpmodels.common.registry import registry
from ocpmodels.models.utils.activations import swish

from torch_geometric.data import Batch

# Implementation of positional encoding obtained from Harvard's annotated transformer's guide
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

@registry.register_model("indfaenet")
class indFAENet(BaseModel): # Change to make it inherit from base model.
    def __init__(self, **kwargs):
        super(indFAENet, self).__init__()

        self.regress_forces = kwargs["regress_forces"]

        self.ads_model = FAENet(**kwargs)
        self.cat_model = FAENet(**kwargs)

        self.act = (
            getattr(nn.functional, kwargs["act"]) if kwargs["act"] != "swish" else swish
        )

        self.disconnected_mlp = kwargs.get("disconnected_mlp", False)
        if self.disconnected_mlp:
            self.ads_lin = Linear(kwargs["hidden_channels"] // 2, kwargs["hidden_channels"] // 2)
            self.cat_lin = Linear(kwargs["hidden_channels"] // 2, kwargs["hidden_channels"] // 2)

        self.transformer_out = kwargs.get("transformer_out", False)
        if self.transformer_out:
            self.combination = Transformer(
                d_model = kwargs["hidden_channels"] // 2,
                nhead = 1,
                num_encoder_layers = 1,
                num_decoder_layers = 1,
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

    def energy_forward(self, data, mode = "train"): # PROBLEM TO FIX: THE PREDICTION IS BY AN AVERAGE!
        batch_size = len(data) // 2

        adsorbates = Batch.from_data_list(data[:batch_size])
        catalysts = Batch.from_data_list(data[batch_size:])

        # Fixing neighbor's dimensions. This error happens when an adsorbate has 0 edges.
        num_adsorbates = len(adsorbates)
        # Find indices of adsorbates without edges:
        edgeless_ads = [
            i for i 
            in range(num_adsorbates) 
            if adsorbates[i].neighbors.shape[0] == 0
        ]
        if len(edgeless_ads) > 0:
            # Since most adsorbates have an edge,
            # we pop those values specifically from range(num_adsorbates)
            mask = list(range(num_adsorbates))
            num_popped = 0 # We can do this since edgeless is already sorted
            for unwanted in edgeless_ads:
                mask.pop(unwanted-num_popped)
                num_popped += 1
            new_nbrs = torch.zeros(
                num_adsorbates,
                dtype = torch.int64,
                device = adsorbates.neighbors.device,
            )
            new_nbrs[mask] = adsorbates.neighbors
            adsorbates.neighbors = new_nbrs

        # Now for catalysts
        num_catalysts = len(catalysts)
        edgeless_cats = [i for i in range(num_catalysts) if catalysts[i].neighbors.shape[0] == 0]
        if len(edgeless_cats) > 0:
            mask = list(range(num_catalysts))
            num_popped = 0
            for unwanted in edgeless_cats:
                mask.pop(unwanted-num_popped)
                num_popped += 1

            # Now, we create the new neighbors. 
            new_nbrs = torch.zeros(
                num_catalysts,
                dtype = torch.int64,
                device = catalysts.neighbors.device,
            )
            new_nbrs[mask] = catalysts.neighbors
            catalysts.neighbors = new_nbrs

        # We make predictions for each
        pred_ads = self.ads_model(adsorbates, mode)
        pred_cat = self.cat_model(catalysts, mode)

        ads_energy = pred_ads["energy"]
        cat_energy = pred_cat["energy"]
        if self.disconnected_mlp:
            ads_energy = self.ads_lin(ads_energy)
            cat_energy = self.cat_lin(cat_energy)

        if self.transformer_out:
            batch_size = ads_energy.shape[0]
            
            fake_target_sequence = self.query_pos.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
            system_energy = torch.cat(
                [
                    ads_energy.unsqueeze(1),
                    cat_energy.unsqueeze(1)
                ],
                dim = 1
            )

            system_energy = self.positional_encoding(system_energy)
            
            system_energy = self.combination(system_energy, fake_target_sequence).squeeze(1)
            system_energy = self.transformer_lin(system_energy)
        else:
            system_energy = torch.cat([ads_energy, cat_energy], dim = 1)
            system_energy = self.combination(system_energy)

        # We combine predictions and return them
        pred_system = {
            "energy" : system_energy,
            "pooling_loss" : pred_ads["pooling_loss"] if pred_ads["pooling_loss"] is None
                else pred_ads["pooling_loss"] + pred_cat["pooling_loss"],
            "hidden_state" : torch.cat([pred_ads["hidden_state"], pred_cat["hidden_state"]], dim = 0)
        }

        return pred_system
