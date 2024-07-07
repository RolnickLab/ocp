"""
Implementation of Ewald message passing by Kosmala et al. (2023)
https://github.com/arthurkosmala/EwaldMP
"""

import numpy as np
import torch
from torch_scatter import scatter

from ocpmodels.models.gemnet.layers.base_layers import Dense, ResidualLayer

from ocpmodels.modules.scaling.scale_factor import ScaleFactor
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis


def get_k_index_product_set(num_k_x, num_k_y, num_k_z):
    # Get a box of k-lattice indices around (0,0,0)
    k_index_sets = (
        torch.arange(-num_k_x, num_k_x + 1, dtype=torch.float),
        torch.arange(-num_k_y, num_k_y + 1, dtype=torch.float),
        torch.arange(-num_k_z, num_k_z + 1, dtype=torch.float),
    )
    k_index_product_set = torch.cartesian_prod(*k_index_sets)
    # Cut the box in half (we will always assume point symmetry)
    k_index_product_set = k_index_product_set[k_index_product_set.shape[0] // 2 + 1 :]

    # Amount of k-points
    num_k_degrees_of_freedom = k_index_product_set.shape[0]

    return k_index_product_set, num_k_degrees_of_freedom


def get_k_voxel_grid(k_cutoff, delta_k, num_k_rbf):
    # Get indices for a cube of k-lattice sites containing the cutoff sphere
    num_k = k_cutoff / delta_k
    k_index_product_set, _ = get_k_index_product_set(num_k, num_k, num_k)

    # Orthogonal k-space basis, norm delta_k
    k_cell = torch.tensor(
        [[delta_k, 0, 0], [0, delta_k, 0], [0, 0, delta_k]], dtype=torch.float
    )

    # Translate lattice indices into k-vectors
    k_grid = torch.matmul(k_index_product_set, k_cell)

    # Prune all k-vectors outside the cutoff sphere
    k_grid = k_grid[torch.sum(k_grid**2, dim=-1) <= k_cutoff**2]

    # Probably quite arbitrary, for backwards compatibility with scaling
    # yaml files produced with old Ewald Message Passing code
    k_offset = 0.1 if num_k_rbf <= 48 else 0.25

    # Evaluate a basis of Gaussian RBF on the k-vectors
    k_rbf_values = RadialBasis(
        num_radial=num_k_rbf,
        # Avoids zero or extremely small RBF values (there are k-points until
        # right at the cutoff, where all RBF would otherwise drop to 0)
        cutoff=k_cutoff + k_offset,
        rbf={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
    )(
        torch.linalg.norm(k_grid, dim=-1)
    )  # Tensor of shape (N_k, N_RBF)

    num_k_degrees_of_freedom = k_rbf_values.shape[-1]

    return k_grid, k_rbf_values, num_k_degrees_of_freedom


def pos_svd_frame(data):
    pos = data.pos
    batch = data.batch
    batch_size = int(batch.max()) + 1

    with torch.cuda.amp.autocast(False):
        rotated_pos_list = []
        for i in range(batch_size):
            # Center each structure around mean position
            pos_batch = pos[batch == i]
            pos_batch = pos_batch - pos_batch.mean(0)

            # Rotate each structure into its SVD frame
            # (only can do this if structure has at least 3 atoms,
            # i.e., the position matrix has full rank)
            if pos_batch.shape[0] > 2:
                U, S, V = torch.svd(pos_batch)
                rotated_pos_batch = torch.matmul(pos_batch, V)

            else:
                rotated_pos_batch = pos_batch

            rotated_pos_list.append(rotated_pos_batch)

        pos = torch.cat(rotated_pos_list)

    return pos


def x_to_k_cell(cells):
    cross_a2a3 = torch.cross(cells[:, 1], cells[:, 2], dim=-1)
    cross_a3a1 = torch.cross(cells[:, 2], cells[:, 0], dim=-1)
    cross_a1a2 = torch.cross(cells[:, 0], cells[:, 1], dim=-1)
    vol = torch.sum(cells[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    b1 = 2 * np.pi * cross_a2a3 / vol
    b2 = 2 * np.pi * cross_a3a1 / vol
    b3 = 2 * np.pi * cross_a1a2 / vol

    bcells = torch.stack((b1, b2, b3), dim=1)

    return (bcells, vol[:, 0])


def get_ewald_params(ewald_hyperparams, use_pbc, hidden_channels):
    if use_pbc:
        # Integer values to define box of k-lattice indices
        num_k_x = ewald_hyperparams["num_k_x"]
        num_k_y = ewald_hyperparams["num_k_y"]
        num_k_z = ewald_hyperparams["num_k_z"]
        delta_k = None
    else:
        k_cutoff = ewald_hyperparams["k_cutoff"]
        # Voxel grid resolution
        delta_k = ewald_hyperparams["delta_k"]
        # Radial k-filter basis size
        num_k_rbf = ewald_hyperparams["num_k_rbf"]
    downprojection_size = ewald_hyperparams["downprojection_size"]
    # Number of residuals in update function
    num_hidden = ewald_hyperparams["num_hidden"]

    k_grid = None
    if use_pbc:
        (
            k_index_product_set,
            num_k_degrees_of_freedom,
        ) = get_k_index_product_set(
            num_k_x,
            num_k_y,
            num_k_z,
        )
        k_rbf_values = None
        delta_k = None
    else:
        # Get the k-space voxel and evaluate Gaussian RBF (can be done at
        # initialization time as voxel grid stays fixed for all structures)
        (
            k_grid,
            k_rbf_values,
            num_k_degrees_of_freedom,
        ) = get_k_voxel_grid(
            k_cutoff,
            delta_k,
            num_k_rbf,
        )

    # Downprojection layer, weights are shared among all interaction blocks
    downproj_layer = Dense(
        num_k_degrees_of_freedom,
        downprojection_size,
        activation=None,
        bias=False,
    )

    return {
        "downproj_layer": downproj_layer,
        "downprojection_size": downprojection_size,
        "num_hidden": num_hidden,
        "hidden_channels": hidden_channels,
        "k_grid": k_grid,
        "activation": "silu",
        "use_pbc": use_pbc,
        "delta_k": delta_k,
        "k_rbf_values": k_rbf_values,
        "k_index_product_set": k_index_product_set,
    }


class EwaldBlock(torch.nn.Module):
    """
    Long-range block from the Ewald message passing method

    Parameters
    ----------
        shared_downprojection: Dense,
            Downprojection block in Ewald block update function,
            shared between subsequent Ewald Blocks.
        emb_size_atom: int
            Embedding size of the atoms.
        downprojection_size: int
            Dimension of the downprojection bottleneck
        num_hidden: int
            Number of residual blocks in Ewald block update function.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
        use_pbc: bool
            Set to True if periodic boundary conditions are applied.
        delta_k: float
            Structure factor voxel resolution
            (only relevant if use_pbc == False).
        k_rbf_values: torch.Tensor
            Pre-evaluated values of Fourier space RBF
            (only relevant if use_pbc == False).
        return_k_params: bool = True,
            Whether to return k,x dot product and damping function values.
    """

    def __init__(
        self,
        shared_downprojection: Dense,
        emb_size_atom: int,
        downprojection_size: int,
        num_hidden: int,
        activation=None,
        name=None,  # identifier in case a ScalingFactor is applied to Ewald output
        use_pbc: bool = True,
        delta_k: float = None,
        k_rbf_values: torch.Tensor = None,
        return_k_params: bool = True,
    ):
        super().__init__()
        self.use_pbc = use_pbc
        self.return_k_params = return_k_params

        self.delta_k = delta_k
        self.k_rbf_values = k_rbf_values

        self.down = shared_downprojection
        self.up = Dense(downprojection_size, emb_size_atom, activation=None, bias=False)
        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )
        self.ewald_layers = self.get_mlp(
            emb_size_atom, emb_size_atom, num_hidden, activation
        )
        if name is not None:
            self.ewald_scale_sum = ScaleFactor(name + "_sum")
        else:
            self.ewald_scale_sum = None

    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(num_hidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        k: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        # Dot products k^Tx and damping values: need to be computed only once per structure
        # Ewald block in first interaction block gets None as input, therefore computes these
        # values and then passes them on to Ewald blocks in later interaction blocks
        dot: torch.Tensor = None,
        sinc_damping: torch.Tensor = None,
    ):
        hres = self.pre_residual(h)
        # Compute dot products and damping values if not already done so by an Ewald block
        # in a previous interaction block
        if dot == None:
            b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])
            dot = torch.sum(torch.gather(k, 0, b) * x.unsqueeze(-2), dim=-1)
        if sinc_damping == None:
            if self.use_pbc == False:
                sinc_damping = (
                    torch.sinc(0.5 * self.delta_k * x[:, 0].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 1].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 2].unsqueeze(-1))
                )
                sinc_damping = sinc_damping.expand(-1, k.shape[-2])
            else:
                sinc_damping = 1

        # Compute Fourier space filter from weights
        if self.use_pbc:
            self.kfilter = (
                torch.matmul(self.up.linear.weight, self.down.linear.weight)
                .T.unsqueeze(0)
                .expand(num_batch, -1, -1)
            )
        else:
            self.k_rbf_values = self.k_rbf_values.to(x.device)
            self.kfilter = (
                self.up(self.down(self.k_rbf_values))
                .unsqueeze(0)
                .expand(num_batch, -1, -1)
            )

        # Compute real and imaginary parts of structure factor
        sf_real = hres.new_zeros(num_batch, dot.shape[-1], hres.shape[-1]).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.cos(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
        )
        sf_imag = hres.new_zeros(num_batch, dot.shape[-1], hres.shape[-1]).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.sin(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
        )

        # Apply Fourier space filter; scatter back to position space
        h_update = 0.01 * torch.sum(
            torch.index_select(sf_real * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.cos(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1])
            + torch.index_select(sf_imag * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.sin(dot))
            .unsqueeze(-1)
            .expand(-1, -1, hres.shape[-1]),
            dim=1,
        )

        if self.ewald_scale_sum is not None:
            h_update = self.ewald_scale_sum(h_update, ref=h)

        # Apply update function
        for layer in self.ewald_layers:
            h_update = layer(h_update)

        if self.return_k_params:
            return h_update, dot, sinc_damping
        else:
            return h_update


# Atom-to-atom continuous-filter convolution
class HadamardBlock(torch.nn.Module):
    """
    Aggregate atom-to-atom messages by Hadamard (i.e., component-wise)
    product of embeddings and radial basis functions

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_bf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "hadamard_atom_update",
    ):
        super().__init__()
        self.name = name

        self.dense_bf = Dense(emb_size_bf, emb_size_atom, activation=None, bias=False)
        self.scale_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum")
        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )
        self.layers = self.get_mlp(emb_size_atom, emb_size_atom, nHidden, activation)

    def get_mlp(self, units_in, units, nHidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, bf, idx_s, idx_t):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]
        h_res = self.pre_residual(h)

        mlp_bf = self.dense_bf(bf)

        x = torch.index_select(h_res, 0, idx_s) * mlp_bf

        x2 = scatter(x, idx_t, dim=0, dim_size=nAtoms, reduce="sum")
        # (nAtoms, emb_size_edge)
        x = self.scale_sum(h, x2)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x
