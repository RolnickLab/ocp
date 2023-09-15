import torch, math
from torch import nn
from torch.nn import Linear

from ocpmodels.models.gemnet.gemnet import GemNetT
from ocpmodels.models.base_model import BaseModel
from ocpmodels.common.registry import registry
from ocpmodels.models.utils.activations import swish

from torch_geometric.data import Batch

@registry.register_model("indgemnet_t")
class indGemNetT(BaseModel): # Change to make it inherit from base model.
    def __init__(self, **kwargs):
        super().__init__()

        self.regress_forces = kwargs["regress_forces"]

        kwargs["num_targets"] = kwargs["emb_size_atom"] // 2

        self.ads_model = GemNetT(**kwargs)
        self.cat_model = GemNetT(**kwargs)

        self.act = swish
        self.combination = nn.Sequential(
            Linear(kwargs["emb_size_atom"] // 2 * 2, kwargs["emb_size_atom"] // 2),
            self.act,
            Linear(kwargs["emb_size_atom"] // 2, 1)
        )

    def energy_forward(self, data, mode = "train"): # PROBLEM TO FIX: THE PREDICTION IS BY AN AVERAGE!
        import ipdb
        ipdb.set_trace()

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
            "E_t": pred_ads["E_t"],
            "idx_t": pred_ads["idx_t"],
            "main_graph": pred_ads["main_graph"],
            "num_atoms": pred_ads["num_atoms"],
            "pos": pred_ads["pos"],
            "F_st": pred_ads["F_st"]
        }

        return pred_system
