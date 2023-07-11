import torch
from torch import nn

from ocpmodels.models.faenet import FAENet
from ocpmodels.models.base_model import BaseModel
from ocpmodels.common.registry import registry

from torch_geometric.data import Batch

@registry.register_model("indfaenet")
class indFAENet(BaseModel): # Change to make it inherit from base model.
    def __init__(self, **kwargs):
        super(indFAENet, self).__init__()

        self.regress_forces = kwargs["regress_forces"]

        self.ads_model = FAENet(**kwargs)
        self.cat_model = FAENet(**kwargs)
        # To do this, you can create a new input to FAENet so that
        # it makes it predict a vector, where the default is normal FAENet.

    def energy_forward(self, data, mode = "train"): # PROBLEM TO FIX: THE PREDICTION IS BY AN AVERAGE!
        batch_size = len(data) // 2

        adsorbates = Batch.from_data_list(data[:batch_size])
        catalysts = Batch.from_data_list(data[batch_size:])

        # Fixing neighbor's dimensions. This error happens when an adsorbate has 0 edges.
        # We do so only on adsorbates, because it is reasonable to assume catalysts have at least one edge.
        num_adsorbates = len(adsorbates)
        # Find indices of adsorbates without edges:
        edgeless = [i for i in range(num_adsorbates) if adsorbates[i].neighbors.shape[0] == 0]
        if len(edgeless) > 0:
            # Since most adsorbates have an edge, we pop those values specifically from range(num_adsorbates)
            mask = list(range(num_adsorbates))
            num_popped = 0 # We can do this since edgeless is already sorted
            for unwanted in edgeless:
                mask.pop(unwanted-num_popped)
                num_popped += 1

            # Now, we create the new neighbors. 
            new_nbrs = torch.zeros(
                num_adsorbates,
                dtype = torch.int64,
                device = adsorbates.neighbors.device,
            )
            new_nbrs[mask] = adsorbates.neighbors
            adsorbates.neighbors = new_nbrs

        # We make predictions for each
        pred_ads = self.ads_model(adsorbates, mode)
        pred_cat = self.cat_model(catalysts, mode)

        # We combine predictions and return them
        pred_system = {
            "energy" : (pred_ads["energy"] + pred_cat["energy"]) / 2,
            "pooling_loss" : pred_ads["pooling_loss"] if pred_ads["pooling_loss"] is None
                else pred_ads["pooling_loss"] + pred_cat["pooling_loss"],
            "hidden_state" : torch.cat([pred_ads["hidden_state"], pred_cat["hidden_state"]], dim = 0)
        }

        return pred_system
