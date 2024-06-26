"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Borrowed from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/transforms/random_rotate.py
# with changes to keep track of the rotation / inverse rotation matrices.

import math
import numbers
import random

import torch
import torch_geometric
from torch_geometric.transforms import LinearTransformation


class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If `degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axes (int, optional): The rotation axes. (default: `[0, 1, 2]`)
    """

    def __init__(self, degrees, axes=[0, 1, 2]):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axes = axes

    def __call__(self, data):
        if data.pos.size(-1) == 2:
            degree = math.pi * random.uniform(*self.degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)
            matrix = [[cos, sin], [-sin, cos]]
        else:
            m1, m2, m3 = torch.eye(3), torch.eye(3), torch.eye(3)
            if 0 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m1 = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
            if 1 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m2 = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
            if 2 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m3 = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

            matrix = torch.mm(torch.mm(m1, m2), m3)

        data_rotated = LinearTransformation(matrix)(data)
        if torch_geometric.__version__.startswith("2."):
            matrix = matrix.T

        # LinearTransformation only rotates `.pos`; need to rotate `.cell` too.
        if hasattr(data_rotated, "cell"):
            data_rotated.cell = torch.matmul(
                data_rotated.cell, matrix.to(data_rotated.cell.device)
            )
        # And .force too
        if hasattr(data_rotated, "force"):
            data_rotated.force = torch.matmul(
                data_rotated.force, matrix.to(data_rotated.force.device)
            )

        return (
            data_rotated,
            matrix,
            torch.inverse(matrix),
        )

    def __repr__(self):
        return "{}({}, axis={})".format(
            self.__class__.__name__, self.degrees, self.axis
        )


class RandomReflect(object):
    r"""Reflect node positions around a specific axis (x, y, x=y) or the origin.
    Take a random reflection type from a list of reflection types.
        (type 0: reflect wrt x-axis, type1: wrt y-axis, type2: y=x, type3: origin)
    """

    def __init__(self):
        self.reflection_type = random.choice([0, 1, 2, 3])

    def __call__(self, data):
        if self.reflection_type == 0:
            matrix = torch.FloatTensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif self.reflection_type == 1:
            matrix = torch.FloatTensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        elif self.reflection_type == 2:
            matrix = torch.FloatTensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        elif self.reflection_type == 3:
            matrix = torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        data_reflected = LinearTransformation(matrix)(data)

        if torch_geometric.__version__.startswith("2."):
            matrix = matrix.T

        # LinearTransformation only rotates `.pos`; need to rotate `.cell` too.
        if hasattr(data_reflected, "cell"):
            data_reflected.cell = torch.matmul(
                data_reflected.cell, matrix.to(data_reflected.cell.device)
            )
        # And .force too
        if hasattr(data_reflected, "force"):
            data_reflected.force = torch.matmul(
                data_reflected.force, matrix.to(data_reflected.force.device)
            )

        return (
            data_reflected,
            matrix,
            torch.inverse(matrix),
        )
