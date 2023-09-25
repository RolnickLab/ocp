import torch, math
from torch import nn
from torch.nn import Linear, Transformer

from ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlus
from ocpmodels.models.base_model import BaseModel
from ocpmodels.common.registry import registry
from ocpmodels.models.utils.activations import swish

from torch_geometric.data import Batch

@registry.register_model("inddpp")
class indDimeNetPlusPlus(BaseModel): # Change to make it inherit from base model.
    def __init__(self, **kwargs):
        super().__init__()

        self.regress_forces = kwargs["regress_forces"]
        kwargs["num_targets"] = kwargs["hidden_channels"] // 2

        import ipdb
        ipdb.set_trace()

        self.cat_model = DimeNetPlusPlus(**kwargs)

        old_hc = kwargs["hidden_channels"]
        old_sphr = kwargs["num_spherical"]
        old_radi = kwargs["num_radial"]
        old_out_emb = kwargs["out_emb_channels"]
        old_targets = kwargs["num_targets"]

        kwargs["hidden_channels"] = kwargs["hidden_channels"] // 2
        kwargs["num_spherical"] = kwargs["num_spherical"] // 2
        kwargs["num_radial"] = kwargs["num_radial"] // 2
        kwargs["out_emb_channesl"] = kwargs["out_emb_channels"] // 2
        kwargs["num_targets"] = kwargs["num_targets"] // 2

        self.ads_model = DimeNetPlusPlus(**kwargs)

        self.act = swish
        self.combination = nn.Sequential(
            Linear(kwargs["num_targets"] + old_targets, kwargs["num_targets"] // 2),
            self.act,
            Linear(kwargs["num_targets"] // 2, 1)
        )

    def energy_forward(self, data, mode = "train"): # PROBLEM TO FIX: THE PREDICTION IS BY AN AVERAGE!
        adsorbates = data[0]
        catalysts = data[1]

        # We make predictions for each
        pred_ads = self.ads_model(adsorbates, mode)
        pred_cat = self.cat_model(catalysts, mode)

        ads_energy = pred_ads["energy"]
        cat_energy = pred_cat["energy"]

        # We combine predictions
        system_energy = torch.cat([ads_energy, cat_energy], dim = 1)
        system_energy = self.combination(system_energy)

        # We return them
        pred_system = {
            "energy" : system_energy,
            "pooling_loss" : pred_ads["pooling_loss"] if pred_ads["pooling_loss"] is None
                else pred_ads["pooling_loss"] + pred_cat["pooling_loss"]
        }

        return pred_system
