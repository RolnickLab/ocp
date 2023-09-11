import torch
from torch.nn import Linear
from torch_scatter import scatter

from ocpmodels.models.gemnet_oc.gemnet_oc import GemNetOC
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad

from torch_geometric.data import Batch

@registry.register_model("depgemnet_oc")
class depGemNetOC(GemNetOC):
    def __init__(self, **kwargs):
        import ipdb
        ipdb.set_trace()
        kwargs["num_targets"] = kwargs["emb_size_atom"] // 2
        super().__init__(**kwargs)
        import ipdb
        ipdb.set_trace()

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
        system = self.sys_lin1(system)
        system = self.sys_lin2(system)

        return system
