import torch
from torch.nn import Linear
from torch_scatter import scatter

from ocpmodels.models.gemnet.gemnet import GemNetT
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, scatter_det

from torch_geometric.data import Batch


@registry.register_model("depgemnet_t")
class depGemNetT(GemNetT):
    def __init__(self, **kwargs):
        self.hidden_channels = kwargs["emb_size_atom"]

        kwargs["num_targets"] = self.hidden_channels // 2
        super().__init__(**kwargs)

        self.sys_lin1 = Linear(self.hidden_channels // 2 * 2, self.hidden_channels // 2)
        self.sys_lin2 = Linear(self.hidden_channels // 2, 1)

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        # We need to save the tags so this step is necessary.
        self.tags_saver(data.tags)
        pred = super().energy_forward(data)

        return pred

    def tags_saver(self, tags):
        self.current_tags = tags

    @conditional_grad(torch.enable_grad())
    def scattering(self, E_t, batch, dim, dim_size, reduce="add"):
        ads = self.current_tags == 2
        cat = ~ads

        ads_out = scatter_det(src=E_t, index=batch * ads, dim=dim, reduce=reduce)
        cat_out = scatter_det(src=E_t, index=batch * cat, dim=dim, reduce=reduce)

        system = torch.cat([ads_out, cat_out], dim=1)
        system = self.sys_lin1(system)
        system = self.sys_lin2(system)

        return system
