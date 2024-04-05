from copy import deepcopy
from ocpmodels.common.graph_transforms import RandomRotate

import torch
from ocpmodels.preprocessing.vn_cano_fct import VNShallowNet
from ocpmodels.preprocessing.vn_pointcloud import VNSmall

from torch_geometric.nn import knn_graph, radius_graph
from ocpmodels.common.utils import get_pbc_distances, radius_graph_pbc, compute_neighbors

def modified_gram_schmidt(vectors): # From Kaba et al. 2023
    v1 = vectors[:, 0]
    v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
    v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
    v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
    v3 = vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
    v3 = v3 - torch.sum(v3 * v2, dim=1, keepdim=True) * v2
    v3 = v3 / torch.norm(v3, dim=1, keepdim=True)

    # if any of the vectors are nan
    # This is due to the fact that the vectors are not linearly independent
    if torch.isnan(v1).any() or torch.isnan(v2).any() or torch.isnan(v3).any():
        # breakpoint()
        vectors = vectors + 1e-8 * torch.randn_like(vectors)
        return modified_gram_schmidt(vectors)

    return torch.stack([v1, v2, v3], dim=1)


def cano_fct_3D(pos, cell, cano_method, edges=None):
    """Computes new positions for the graph atoms using PCA

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph
        cano_method (str): canonicalisation method used (currently not used)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        tensor: updated atom positions
        tensor: updated unit cell
        tensor: the rotation matrix used
    """
    if cano_method is not None:
        pass

    vn_cell = deepcopy(cell)
    vn_pos = deepcopy(pos)

    # vn_model = VNShallowNet(
    #     in_dim=1, # Only positions
    #     out_dim=4, # rotation (3) + translation (1)
    # )
    # if edges is None: # Compute k-nearest neighbors graph
    #     k = 2  # Number of neighbors
    #     pos_tensor = vn_pos.clone().detach()
    #     edges = knn_graph(pos_tensor, k, batch=None, loop=False)
    # edges = torch.tensor([])

    vn_model = VNSmall()

    if not torch.is_grad_enabled():
        print("\nWarning: enabling in preprocessing.\n")
        torch.set_grad_enabled(True) 
    vn_rot = vn_model(vn_pos)
    
    vn_rot = modified_gram_schmidt(vn_rot)

    vn_cell = vn_cell @ vn_rot
    vn_pos = vn_pos @ vn_rot
    
    return [vn_pos.squeeze()], [vn_cell], [vn_rot]



def cano_fct_2D(pos, cell, cano_method, edges=None):
    """Computes new positions for the graph atoms,
    based on a frame averaging building on PCA.

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph
        cano_method (str): canonicalisation method used (currently not used)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        tensor: updated atom positions
        tensor: updated unit cell
        tensor: the rotation matrix used
    """
    # Exit with error because not implemented
    raise NotImplementedError("2D LCF is not implemented yet.")

    if cano_method is not None:
        pass

    # Compute transformations
    vn_pos, vn_cell, vn_rot = compute_frames(pos, cell, edges=edges)
    return vn_pos, vn_cell, vn_rot


def data_augmentation(g, d=3, *args):
    """Data augmentation where we randomly rotate each graph
    in the dataloader transform

    Args:
        g (data.Data): single graph
        d (int): dimension of the DA rotation (2D around z-axis or 3D)
        rotation (str, optional): around which axis do we rotate it.
            Defaults to 'z'.

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    if d == 3:
        transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D
    else:
        transform = RandomRotate([-180, 180], [2])  # 2D around z-axis

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated