import torch
from torch import nn
from torch.nn import Linear
from torch_scatter import scatter

from ocpmodels.models.schnet import SchNet
from ocpmodels.models.faenet import OutputBlock as conOutputBlock
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.utils.activations import swish

from torch_geometric.data import Batch

@registry.register_model("depschnet")
class depSchNet(SchNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We replace the last linear transform to keep dimentionality
        self.lin2 = Linear(self.hidden_channels // 2, self.hidden_channels // 2)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

        self.combination = nn.Sequential(
            Linear(self.hidden_channels // 2 * 2, self.hidden_channels // 2),
            swish,
            Linear(self.hidden_channels // 2, 1)
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
    def scattering(self, h, batch):
        ads = self.current_tags == 2
        cat = ~ads

        ads_out = scatter(h, batch * ads, dim = 0, reduce = self.readout)
        cat_out = scatter(h, batch * cat, dim = 0, reduce = self.readout)

        system = torch.cat([ads_out, cat_out], dim = 1)
        system = self.combination(system)

        return system
