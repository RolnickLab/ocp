import torch

import ocpmodels.preprocessing.frame_averaging as frame_averaging

from ocpmodels.preprocessing.graph_rewiring import (
    one_supernode_per_atom_type,
    one_supernode_per_atom_type_dist,
    one_supernode_per_graph,
    remove_tag0_nodes,
)

import ocpmodels.preprocessing.trained_cano as trained_cano

from ocpmodels.preprocessing.vn_pointcloud import VNSmall, VNPointnet, VN_dgcnn


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


class BaseUntrainableCanonicalisation(Transform):
    r"""
    Base Untrainable Canonicalisation functions for (PyG) Data objects (e.g. 3D atomic graphs).
    Args:
        equivariance_module: which equivariance module to use, can be "fa" or "untrained_cano"
        cano_type: "3D", "2D", "DA" or "" (no equivariance imposed)
    """

    def __init__(self, cano_args=None):
        self.equivariance_module = cano_args.get("equivariance_module", "fa")
        self.cano_type = cano_args.get("cano_type", "")

        if self.equivariance_module == "fa":
            self.equivariance_module = FrameAveraging(**cano_args)
        elif self.equivariance_module == "untrained_cano":
            self.equivariance_module = UntrainedCanonicalisation(**cano_args)
        else:  # No untrained canonicalisation used
            self.equivariance_module = FrameAveraging(cano_type=None, fa_method=None)

    def __call__(self, data):
        if type(self.equivariance_module) == str:
            return data
        return self.equivariance_module.call(data)


class BaseTrainableCanonicalisation(Transform):
    r"""
    Base Trainable Canonicalisation functions for (PyG) Data objects (e.g. 3D atomic graphs).
    Args:
        equivariance_module: which equivariance module to use, can be "trained_cano"
        cano_type: "3D", "2D", "DA" or "" (no equivariance imposed)
    """

    def __init__(self, cano_model, cano_args=None):
        self.equivariance_module = cano_args.get("equivariance_module", "fa")
        self.cano_type = cano_args.get("cano_type", "")

        if self.equivariance_module == "trained_cano":
            self.equivariance_module = TrainedCanonicalisation(cano_model, **cano_args)
        else:  # No trainable canonicalisation used
            self.equivariance_module = FrameAveraging(cano_type=None, fa_method=None)

    def __call__(self, data):
        if type(self.equivariance_module) == str:
            return data
        return self.equivariance_module.call(data)


class UntrainedCanonicalisation:
    r"""Untrained canonicalisation functions for (PyG) Data objects (e.g. 3D atomic graphs).
    Args:
        cano_type (str):
            Can be 2D, 3D, Data Augmentation or no equivariance imposed, respectively denoted
            by (`"2D"`, `"3D"`, `"DA"`, `""`)
        cano_method (str): currently not used, in case several canonicalisation methods are
        implemented.

    Returns:
        (data.Data): updated data object with new positions (+ unit cell) attributes
        and the rotation matrices used for the frame averaging transform.
    """

    def __init__(self, cano_type=None, cano_method=None, **kwargs):
        self.cano_method = (
            "default" if (cano_method is None or cano_method == "") else cano_method
        )
        self.cano_type = "" if cano_type is None else cano_type
        self.inactive = not self.cano_type
        assert self.cano_type in {
            "",
            "2D",
            "3D",
            "DA",
        }

        self.cano_model = get_learnable_model(cano_method)

        for param in self.cano_model.parameters():
            param.requires_grad = False

        if self.cano_type:
            if self.cano_type == "2D":
                self.cano_func = (
                    trained_cano.cano_fct_3D
                )  # To be changed if 2D becomes implemented
            elif self.cano_type == "3D":
                self.cano_func = trained_cano.cano_fct_3D
            elif self.cano_type == "DA":
                self.cano_func = trained_cano.data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.cano_type}")

    def call(self, data):
        if self.inactive:
            return data
        elif self.cano_type == "DA":
            return self.cano_func(data, self.cano_method)
        else:
            data.cano_pos, data.cano_cell, data.cano_rot = self.cano_func(
                self.cano_model,
                data.pos,
                data.cell if hasattr(data, "cell") else None,
                self.cano_method,
                data.edge_index if hasattr(data, "edge_index") else None,
            )
            return data


class TrainedCanonicalisation:
    r"""Trained canonicalisation functions for (PyG) Data objects (e.g. 3D atomic graphs).
    Args:
        cano_type (str):
            Can be 2D, 3D, Data Augmentation or no equivariance imposed, respectively denoted
            by (`"2D"`, `"3D"`, `"DA"`, `""`)
        cano_method (str): currently not used, in case several canonicalisation methods are
        implemented.

    Returns:
        (data.Data): updated data object with new positions (+ unit cell) attributes
        and the rotation matrices used for the frame averaging transform.
    """

    def __init__(self, cano_model, cano_type=None, cano_method=None, **kwargs):
        self.cano_method = (
            "default" if (cano_method is None or cano_method == "") else cano_method
        )
        self.cano_type = "" if cano_type is None else cano_type
        self.inactive = not self.cano_type
        assert self.cano_type in {
            "",
            "2D",
            "3D",
            "DA",
        }

        self.cano_model = cano_model

        if self.cano_type:
            if self.cano_type == "2D":
                self.cano_func = (
                    trained_cano.cano_fct_3D
                )  # To be changed if 2D becomes implemented
            elif self.cano_type == "3D":
                self.cano_func = trained_cano.cano_fct_3D
            elif self.cano_type == "DA":
                self.cano_func = trained_cano.data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.cano_type}")

    def call(self, data):
        if self.inactive:
            return data
        elif self.cano_type == "DA":
            return self.cano_func(data, self.cano_method)
        else:
            data.cano_pos, data.cano_cell, data.cano_rot = self.cano_func(
                self.cano_model,
                data.pos,
                data.cell if hasattr(data, "cell") else None,
                self.cano_method,
                data.edge_index if hasattr(data, "edge_index") else None,
            )
            return data


class FrameAveraging:
    r"""Frame Averaging (FA) Transform for (PyG) Data objects (e.g. 3D atomic graphs).
    Computes new atomic positions (`fa_pos`) for all datapoints, as well as new unit
    cells (`fa_cell`) attributes for crystal structures, when applicable. The rotation
    matrix (`fa_rot`) used for the frame averaging is also stored.

    Args:
        cano_type (str): Transform method used.
            Can be 2D FA, 3D FA, Data Augmentation or no FA, respectively denoted by
            (`"2D"`, `"3D"`, `"DA"`, `""`)
        fa_method (str): the actual frame averaging technique used.
            "stochastic" refers to sampling one frame at random (at each epoch), "det"
            to chosing deterministically one frame, and "all" to using all frames. The
            prefix "se3-" refers to the SE(3) equivariant version of the method. ""
            means that no frame averaging is used. (`""`, `"stochastic"`, `"all"`,
            `"det"`, `"se3-stochastic"`, `"se3-all"`, `"se3-det"`)

    Returns:
        (data.Data): updated data object with new positions (+ unit cell) attributes
        and the rotation matrices used for the frame averaging transform.
    """

    def __init__(self, cano_type=None, fa_method=None, **kw_args):
        self.fa_method = (
            "random" if (fa_method is None or fa_method == "") else fa_method
        )
        self.cano_type = "" if cano_type is None else cano_type
        self.inactive = not self.cano_type
        assert self.cano_type in {
            "",
            "2D",
            "3D",
            "DA",
        }
        assert self.fa_method in {
            "",
            "random",
            "det",
            "all",
            "se3-random",
            "se3-det",
            "se3-all",
        }

        if self.cano_type:
            if self.cano_type == "2D":
                self.fa_func = frame_averaging.frame_averaging_2D
            elif self.cano_type == "3D":
                self.fa_func = frame_averaging.frame_averaging_3D
            elif self.cano_type == "DA":
                self.fa_func = frame_averaging.data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.cano_type}")

    def call(self, data):
        if self.inactive:
            return data
        elif self.cano_type == "DA":
            return self.fa_func(data, self.fa_method)
        else:
            data.cano_pos, data.cano_cell, data.cano_rot = self.fa_func(
                data.pos, data.cell if hasattr(data, "cell") else None, self.fa_method
            )
            return data


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
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        if isinstance(data.natoms, int) or data.natoms.ndim == 0:
            data.natoms = torch.tensor([data.natoms])
        if not hasattr(data, "ptr") or data.ptr is None:
            data.ptr = torch.tensor([0, data.natoms])

        return self.rewiring_func(data)


class Compose:
    # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

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


def get_learnable_model(cano_method):
    if cano_method == "pointnet":
        return VNPointnet()
    elif cano_method == "dgcnn":
        return VN_dgcnn()
    elif cano_method == "simple":
        return VNSmall()
    else:
        raise ValueError(f"Unknown canonicalisation method: {cano_method}")


# Both will be called, but in different places
def get_transforms(trainer_config):
    transforms = [
        AddAttributes(),
        GraphRewiring(trainer_config.get("graph_rewiring")),
        BaseUntrainableCanonicalisation(trainer_config.get("cano_args", {})),
    ]
    return Compose(transforms)


def get_learnable_transforms(cano_model, trainer_config):
    transforms = [
        BaseTrainableCanonicalisation(cano_model, trainer_config["cano_args"]),
    ]
    return Compose(transforms)
