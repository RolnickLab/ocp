import torch
from torch.nn import Linear
from torch_scatter import scatter

from ocpmodels.models.faenet import FAENet
from ocpmodels.models.faenet import OutputBlock as conOutputBlock
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad

from torch_geometric.data import Batch

class discOutputBlock(conOutputBlock):
    def __init__(self, energy_head, hidden_channels, act, disconnected_mlp = False):
        super(discOutputBlock, self).__init__(
            energy_head, hidden_channels, act
        )

        self.lin2 = Linear(hidden_channels // 2, hidden_channels // 2)

        self.disconnected_mlp = disconnected_mlp
        if self.disconnected_mlp:
            self.ads_lin = Linear(hidden_channels // 2, hidden_channels // 2)
            self.cat_lin = Linear(hidden_channels // 2, hidden_channels // 2)

        self.sys_lin1 = Linear(hidden_channels // 2 * 2, hidden_channels // 2)
        self.sys_lin2 = Linear(hidden_channels // 2, 1)

    def tags_saver(self, tags):
        self.current_tags = tags

    def forward(self, h, edge_index, edge_weight, batch, alpha):
        if self.energy_head == "weighted-av-final-embeds": # Right now, this is the only available option.
            alpha = self.w_lin(h)

        elif self.energy_head == "graclus":
            h, batch = self.graclus(h, edge_index, edge_weight, batch)

        elif self.energy_head in {"pooling", "random"}:
            h, batch, pooling_loss = self.hierarchical_pooling(
                h, edge_index, edge_weight, batch
            )

        # MLP
        h = self.lin1(h)
        h = self.lin2(self.act(h))

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        ads = self.current_tags == 2
        cat = ~ads

        ads_out = scatter(h, batch * ads, dim = 0, reduce = "add")
        cat_out = scatter(h, batch * cat, dim = 0, reduce = "add")

        if self.disconnected_mlp:
            ads_out = self.ads_lin(ads_out)
            cat_out = self.cat_lin(cat_out)

        system = torch.cat([ads_out, cat_out], dim = 1)

        system = self.sys_lin1(system)
        energy = self.sys_lin2(system)

        return energy

@registry.register_model("depfaenet")
class depFAENet(FAENet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "disconnected_mlp" not in kwargs:
            kwargs["disconnected_mlp"] = False

        self.output_block = discOutputBlock(
            self.energy_head, kwargs["hidden_channels"], self.act, kwargs["disconnected_mlp"]
        )

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        self.output_block.tags_saver(data.tags)
        pred = super().energy_forward(data)

        return pred
