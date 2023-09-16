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

        self.ads_model = DimeNetPlusPlus(**kwargs)
        self.cat_model = DimeNetPlusPlus(**kwargs)

        self.act = swish
        self.combination = nn.Sequential(
            Linear(kwargs["hidden_channels"] // 2 * 2, kwargs["hidden_channels"] // 2),
            self.act,
            Linear(kwargs["hidden_channels"] // 2, 1)
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
