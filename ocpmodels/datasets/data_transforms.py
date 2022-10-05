import torch

from ocpmodels.preprocessing.data_augmentation import (
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
        self.fa_frames = ("all" if fa_frames is None else fa_frames).lower()
        self.fa_type = ("" if fa_type is None else fa_type).lower()
        self.inactive = not self.fa_type
        assert self.fa_type in {
            "",
            "2d",
            "3d",
            "da",
        }
        assert self.fa_frames in {
            "random",
            "det",
            "all",
            "se3-random",
            "se3-det",
            "se3-all",
        }

        if self.fa_type:
            if self.fa_type.lower() == "2d":
                self.fa_func = frame_averaging_2D
            elif self.fa_type.lower() == "3d":
                self.fa_func = frame_averaging_3D
            elif self.fa_type.lower() == "da":
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
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        data.natoms = torch.tensor([data.natoms])
        data.ptr = torch.tensor([0, data.natoms])

        if self.inactive:
            return data
        return self.rewiring_func(data)


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


def get_transforms(trainer_config):
    transforms = [
        GraphRewiring(trainer_config["graph_rewiring"]),
        FrameAveraging(trainer_config["frame_averaging"], trainer_config["fa_frames"]),
    ]
    return Compose(transforms)