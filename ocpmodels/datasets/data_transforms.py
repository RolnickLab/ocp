import torch

from ocpmodels.preprocessing.frame_averaging import (
    data_augmentation,
    frame_averaging_2D,
    frame_averaging_3D,
)
from ocpmodels.preprocessing.graph_rewiring import (
    one_supernode_per_atom_type,
    one_supernode_per_atom_type_dist,
    one_supernode_per_graph,
    remove_tag0_nodes,
)


class Transform:
    def __call__(self, data):
        raise NotImplementedError

    def __str__(self):
        name = self.__class__.__name__
        items = [
            f"{k}={v}"
            for k, v in self.__dict__.items()
            if not callable(v) and k != "inactive"
        ]
        s = f"{name}({', '.join(items)})"
        if self.inactive:
            s = f"[inactive] {s}"
        return s


class FrameAveraging(Transform):
    def __init__(self, fa_type=None, fa_frames=None):
        self.fa_frames = (
            "random" if (fa_frames is None or fa_frames == "") else fa_frames
        )
        self.fa_type = "" if fa_type is None else fa_type
        self.inactive = not self.fa_type
        assert self.fa_type in {
            "",
            "2D",
            "3D",
            "DA",
        }
        assert self.fa_frames in {
            "",  # equivalent to random, necessary still for sweeps
            "random",
            "det",
            "all",
            "multiple",
            "se3-random",
            "se3-det",
            "se3-all",
            "se3-multiple",
        }

        if self.fa_type:
            if self.fa_type == "2D":
                self.fa_func = frame_averaging_2D
            elif self.fa_type == "3D":
                self.fa_func = frame_averaging_3D
            elif self.fa_type == "DA":
                self.fa_func = data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.fa_type}")

    def __call__(self, data):
        if self.inactive:
            return data
        return self.fa_func(data, self.fa_frames)


class GraphRewiring(Transform):
    def __init__(self, rewiring_type=None) -> None:
        self.rewiring_type = rewiring_type

        self.inactive = not self.rewiring_type

        if self.rewiring_type:
            if self.rewiring_type == "remove-tag-0":
                self.rewiring_func = remove_tag0_nodes
            elif self.rewiring_type == "one-supernode-per-graph":
                self.rewiring_func = one_supernode_per_graph
            elif self.rewiring_type == "one-supernode-per-atom-type":
                self.rewiring_func = one_supernode_per_atom_type
            elif self.rewiring_type == "one-supernode-per-atom-type-dist":
                self.rewiring_func = one_supernode_per_atom_type_dist
            else:
                raise ValueError(f"Unknown self.graph_rewiring {self.graph_rewiring}")

    def __call__(self, data):
        if self.inactive:
            return data

        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        data.natoms = torch.tensor([data.natoms])
        data.ptr = torch.tensor([0, data.natoms])

        return self.rewiring_func(data)

class Disconnected(Transform):
    def __init__(self, is_disconnected=False) -> None:
        self.inactive = not is_disconnected

    def edge_classifier(self, edge_index, tags):
        edges_with_tags = tags[edge_index.type(torch.long)] # Tensor with shape=edge_index.shape where every entry is a tag        
        filt1 = (edges_with_tags[0] == edges_with_tags[1])
        filt2 = (edges_with_tags[0] != 2) * (edges_with_tags[1] != 2)

        # Edge is removed if tags are different (R1), and at least one end has tag 2 (R2). We want ~(R1*R2) = ~R1+~R2.
        # filt1 = ~R1. Let L1 be that head has tag 2, and L2 is that tail has tag 2. Then R2 = L1+L2, so ~R2 = ~L1*~L2 = filt2.

        return filt1 + filt2

    def __call__(self, data):
        if self.inactive:
            return data

        values = self.edge_classifier(data.edge_index, data.tags)
        
        data.edge_index = data.edge_index[:, values]
        data.cell_offsets = data.cell_offsets[values, :]
        data.distances = data.distances[values]

        return data
        

class Compose:
    # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class AddAttributes:
    def __call__(self, data):
        if (
            not hasattr(data, "distances")
            and hasattr(data, "edge_index")
            and data.edge_index is not None
        ):
            data.distances = torch.sqrt(
                (
                    (data.pos[data.edge_index[0, :]] - data.pos[data.edge_index[1, :]])
                    ** 2
                ).sum(-1)
            ).float()
        return data


def get_transforms(trainer_config):
    transforms = [
        AddAttributes(),
        GraphRewiring(trainer_config.get("graph_rewiring")),
        FrameAveraging(trainer_config["frame_averaging"], trainer_config["fa_frames"]),
        Disconnected(trainer_config["is_disconnected"])
    ]
    return Compose(transforms)
