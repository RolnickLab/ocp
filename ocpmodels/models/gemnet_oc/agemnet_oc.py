import torch, math
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Data, Batch

from ocpmodels.models.gemnet_oc.gemnet_oc import GemNetOC
from ocpmodels.models.base_model import BaseModel
from ocpmodels.common.registry import registry
from ocpmodels.models.utils.activations import swish

from torch_geometric.data import Batch


@registry.register_model("agemnet_oc")
class aGemNetOC(BaseModel):  # Change to make it inherit from base model.
    def __init__(self, **kwargs):
        super().__init__()

        self.regress_forces = kwargs["regress_forces"]
        self.direct_forces = kwargs["direct_forces"]

        self.regress_forces = kwargs["regress_forces"]

        kwargs["num_targets"] = kwargs["emb_size_atom"] // 2

        self.ads_model = GemNetOC(**kwargs)
        self.cat_model = GemNetOC(**kwargs)

        self.act = swish
        self.combination = nn.Sequential(
            Linear(kwargs["emb_size_atom"] // 2 * 2, kwargs["emb_size_atom"] // 2),
            self.act,
            Linear(kwargs["emb_size_atom"] // 2, 1),
        )

    def energy_forward(
        self, data, mode="train"
    ):  # PROBLEM TO FIX: THE PREDICTION IS BY AN AVERAGE!
        import ipdb

        ipdb.set_trace()

        bip_edges = data["is_disc"].edge_index
        bip_weights = data["is_disc"].edge_weight

        adsorbates, catalysts = [], []
        for i in range(len(data)):
            adsorbates.append(
                Data(
                    **data[i]["adsorbate"]._mapping,
                    edge_index=data[i]["adsorbate", "is_close", "adsorbate"]
                )
            )
            catalyst.append(
                Data(
                    **data[i]["catalyst"]._mapping,
                    edge_index=data[i]["catalyst", "is_close", "catalyst"]
                )
            )
        del data
        adsorbates = Batch.from_data_list(adsorbates)
        catalysts = Batch.from_data_list(catalysts)

        # We make predictions for each
        pos_ads = adsorbates.pos
        batch_ads = adsorbates.batch
        atomic_numbers_ads = adsorbates.atomic_numbers.long()
        num_atoms_ads = adsorbates.shape[0]

        pos_cat = catalysts.pos
        batch_cat = catalysts.batch
        atomic_numbers_cat = catalysts.atomic_numbers.long()
        num_atoms_cat = catalysts.shape[0]

        if self.regress_forces and not self.direct_forces:
            pos_ads.requires_grad_(True)
            pos_cat.requires_grad_(True)

        output_ads = self.ads_model.pre_interaction(
            pos_ads, batch_ads, atomic_numbers_ads, num_atoms_ads, adsorbates
        )
        output_cat = self.cat_model.pre_interaction(
            pos_cat, batch_cat, atomic_numbers_cat, num_atoms_cat, catalysts
        )

        inter_outputs_ads, inter_outputs_cat = self.interactions(output_ads, output_cat)

        ads_energy = pred_ads["energy"]
        cat_energy = pred_cat["energy"]

        # We combine predictions
        system_energy = torch.cat([ads_energy, cat_energy], dim=1)
        system_energy = self.combination(system_energy)

        # We return them
        pred_system = {
            "energy": system_energy,
            "pooling_loss": pred_ads["pooling_loss"]
            if pred_ads["pooling_loss"] is None
            else pred_ads["pooling_loss"] + pred_cat["pooling_loss"],
        }

        return pred_system

    def interactions(self, output_ads, output_cat):
        h_ads, m_ads = output_ads["h"], output_ads["m"]
        h_cat, m_cat = output_cat["h"], output_cat["m"]
        del output_ads["h"]
        del output_ads["m"]
        del output_cat["h"]
        del output_cat["m"]

        # basis_output_ads, idx

        return 1, 2

        # GOT UP TO HERE. I NEED TO DO INTERACTIONS. HERE.
