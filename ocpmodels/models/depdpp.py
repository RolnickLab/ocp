import torch
from torch import nn
from torch.nn import Linear
from torch_scatter import scatter

from ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlus
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.utils.activations import swish

from torch_geometric.data import Batch


@registry.register_model("depdpp")
class depSchNet(DimeNetPlusPlus):
    def __init__(self, **kwargs):
        self.hidden_channels = kwargs["hidden_channels"]

        kwargs["num_targets"] = kwargs["hidden_channels"] // 2
        super().__init__(**kwargs)

        self.act = swish
        self.combination = nn.Sequential(
            Linear(self.hidden_channels // 2 * 2, self.hidden_channels // 2),
            self.act,
            Linear(self.hidden_channels // 2, 1),
        )

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        # We need to save the tags so this step is necessary.
        self.tags_saver(data.tags)
        pred = super().energy_forward(data)

        return pred

    def tags_saver(self, tags):
        self.current_tags = tags

    @conditional_grad(torch.enable_grad())
    def scattering(self, batch, h, P_bis):
        ads = self.current_tags == 2
        cat = ~ads

        ads_out = scatter(h, batch * ads, dim=0)
        cat_out = scatter(h, batch * cat, dim=0)

        system = torch.cat([ads_out, cat_out], dim=1)
        system = self.combination(system)
        system = system + P_bis

        return system
