import sys
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import torch_geometric
import networkx as nx

from ocpmodels.common.flags import flags
from ocpmodels.common.utils import build_config, resolve, setup_imports, merge_dicts
from ocpmodels.trainers.single_trainer import SingleTrainer
from ocpmodels.common.utils import conditional_grad, get_pbc_distances
from ocpmodels.datasets.lmdb_dataset import data_list_collater

runs_dir = "$SCRATCH/ocp/runs"


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output

    return hook


def load_checkpoint(job_id):
    path = resolve(runs_dir) / job_id

    checkpoint = path / "checkpoints" / "best_checkpoint.pt"
    setup_imports()
    argv = deepcopy(sys.argv)
    # trainer_args = flags.parser.parse_args()
    trainer_args = flags.parser.parse_args([])
    sys.argv[1:] = argv
    trainer_args.continue_from_dir = str(path)
    config = build_config(trainer_args, [])
    config["logger"] = "dummy"
    config["checkpoint"] = str(checkpoint)
    config["checkpoint_dir"] = str(path / "checkpoints")
    config["optim"]["batch_size"] = 1
    config["optim"]["eval_batch_size"] = 1
    print(config)

    trainer = SingleTrainer(**config)
    trainer.config["checkpoint_dir"] = str(path / "checkpoints")
    trainer.load_checkpoint(checkpoint)

    return trainer


class GaussianSmearing(nn.Module):
    r"""Smears a distance distribution by a Gaussian function."""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def process_datapoint(data, trainer=None):
    # data = data_list_collater([x]) # if input is just a single graph rather than a batch

    z = data.atomic_numbers.long()
    pos = data.pos
    energy_skip_co = []

    out = get_pbc_distances(
        pos,
        data.edge_index,
        data.cell,
        data.cell_offsets,
        data.neighbors,
        return_distance_vec=True,
    )

    edge_index = out["edge_index"]
    edge_weight = out["distances"]
    rel_pos = out["distance_vec"]
    if trainer is not None:
        distance_expansion = GaussianSmearing(
            0.0, trainer.model.module.cutoff, trainer.model.module.num_gaussians
        )
    edge_attr = distance_expansion(edge_weight)

    new_data = torch_geometric.data.Data(
        x=z,
        pos=pos,  # take the old positions back
        # batch=batch,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell=data.cell,
        cell_offsets=data.cell_offsets,
        neighbors=data.neighbors,
        distance_vec=rel_pos,
        energy_skip_co=energy_skip_co,
        tags=data.tags,
        force=data.force,
    )

    return new_data


def oc20_to_graph(x, processed=True):
    # Undirected graph
    if not processed:
        return to_networkx(
            x,
            node_attrs=["pos", "tags", "force"],
            edge_attrs=["cell_offsets"],
            to_undirected=True,
        )
    return to_networkx(
        x,
        node_attrs=["pos", "tags", "x", "force"],
        edge_attrs=["distance_vec", "cell_offsets"],
        to_undirected=True,
    )


def plot_element_3d(data_graph, order=None, clusters=None):
    if order is None:
        order = list(data_graph.nodes())
    if clusters is not None:
        tags = clusters
    else:
        tags = np.array(list(nx.get_node_attributes(data_graph, "tags").values()))
    print(data_graph)
    data_graph = data_graph.subgraph(order)
    pos = nx.get_node_attributes(data_graph, "pos")
    print(pos)
    # use tags for coloring
    node_xyz = np.array([pos[i] for i in range(len(pos))])
    edge_xyz = np.array([[pos[i], pos[j]] for i, j in data_graph.edges()])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*node_xyz.T, s=50, ec="w", c=tags, cmap="tab20", alpha=1)

    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, c="k", lw=0.5)

    fig.tight_layout()
    plt.show()

def plot_noisy_nodes_target_3d(data):
    pos = data.pos
    pos_relaxed = data.pos_relaxed

    target = pos_relaxed - pos
    if isinstance(target, torch.Tensor):
        target = target.numpy()

    xs, ys, zs = target.T 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=100, c='blue', edgecolor='k', alpha=0.6)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Target Position Predictions')

    plt.tight_layout()
    plt.show()

def plot_atom_positions(positions):
    """
    Plots a graph of atoms with their positions using NetworkX and matplotlib with no edges.

    Args:
    positions_tensor (torch.Tensor or np.ndarray): A 2D tensor or array of shape (N, 3) where N is the number of atoms
    and the columns represent the x, y, and z coordinates of each atom.
    """
    # Ensure the input is a NumPy array
    try:
        positions = positions.detach().cpu().numpy()
    except:
        positions = positions[0].detach().cpu().numpy()

    # Create a graph
    G = nx.Graph()

    # Add nodes with position attributes
    for i, position in enumerate(positions):
        G.add_node(i, pos=position)

    # Extract positions in a format that NetworkX can use for plotting
    pos = nx.get_node_attributes(G, 'pos')

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for nodes
    xs, ys, zs = np.array(list(pos.values())).T
    ax.scatter(xs, ys, zs, s=100, c='blue', edgecolor='k', alpha=0.6)

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Atom Positions in 3D Space')

    # Layout adjustment
    plt.tight_layout()
    plt.show()

def plot_combined_atom_positions(pred_positions, data, title=''):
    """
    Plots predicted atom positions alongside their target positions from a graph in 3D on the same plot,
    with different colors for each and labels showing atom indices. Connects corresponding predicted and target 
    positions with edges.

    Args:
    pred_positions (np.ndarray or torch.Tensor): Predicted positions of atoms as a 2D array or tensor.
    data (object with attributes): An object that has attributes 'pos' and 'pos_relaxed' for positions.
    title (str): Custom title for the plot.
    """
    title = f"Predicted vs Target Pos {title}"
    # Ensure input is a numpy array if it's a tensor
    if isinstance(pred_positions, torch.Tensor):
        pred_positions = pred_positions.detach().cpu().numpy()

    # Calculate target positions
    pos = data.pos
    pos_relaxed = data.pos_relaxed
    target = pos_relaxed - pos
    if isinstance(target, torch.Tensor):
        target = target.numpy()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for predicted positions
    xs, ys, zs = pred_positions.T
    scatter1 = ax.scatter(xs, ys, zs, s=100, c='blue', edgecolor='k', alpha=0.6, label='Predicted')

    # Scatter plot for target positions
    xt, yt, zt = target.T
    scatter2 = ax.scatter(xt, yt, zt, s=100, c='red', edgecolor='k', alpha=0.6, label='Target')

    # Connect corresponding nodes with edges
    for i in range(len(xs)):
        ax.plot([xs[i], xt[i]], [ys[i], yt[i]], [zs[i], zt[i]], 'gray', alpha=0.5)  # Add edge

        # Annotate each point with its index
        ax.text(xs[i], ys[i], zs[i], f'{i}', color='black', size=10)
        ax.text(xt[i], yt[i], zt[i], f'{i}', color='black', size=10)

    # Set labels and custom title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Layout adjustment for better visualization
    plt.tight_layout()
    plt.show()

def print_model_modules(model, indent=0):
    """
    Prints the modules of a given model with structured formatting, using indentation to show hierarchy.
    Also displays the number of trainable parameters and dimensions of linear layers.

    Args:
    model (torch.nn.Module): The model whose modules you want to print.
    indent (int): The indentation level to use for this layer of modules (used internally in recursion).
    """
    for name, module in model.named_children():
        print(' ' * indent + f'{name} -> {module.__class__.__name__}', end='')
        if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
            print(f', Parameters: {module.weight.numel()}', end='')
            if isinstance(module, nn.Linear):
                print(f', In features: {module.in_features}, Out features: {module.out_features}', end='')
        print()  # Move to the next line after printing details about the current module
        # Recursively print child modules
        if list(module.children()):
            print_model_modules(module, indent + 4)  # Increase indentation for child modules
