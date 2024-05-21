import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.utils import make_script_trainer
from ocpmodels.trainers import SingleTrainer
from torch_geometric.data import Batch

if __name__ == "__main__":
    config = {}

    # Customize args
    config["graph_rewiring"] = "remove-tag-0"
    config["frame_averaging"] = "3D"
    config["fa_method"] = "all"
    config["test_ri"] = False
    # config["optim"] = {"batch_size": 1}

    str_args = sys.argv[1:]
    if all("config" not in arg for arg in str_args):
        str_args.append("--is_debug")
        str_args.append("--config=faenet-is2re-10k")

    # Create trainer
    trainer: SingleTrainer = make_script_trainer(str_args=str_args, overrides=config)

    for batch in trainer.loaders["train"]:
        break
    b = batch[0]
    rotated_b = b.clone()
    rotated_b = trainer.rotate_graph(rotated_b, rotation="z")
    rotation_matrix = rotated_b["rot"]
    rotated_b = rotated_b["batch_list"][0]

    # Check: X' = X R (or X = X' R^T)
    assert torch.allclose(rotated_b[0].pos @ rotation_matrix.T, b[0].pos, atol=1e-04)
    assert torch.allclose(b[0].pos @ rotation_matrix, rotated_b[0].pos, atol=1e-04)
    # Check: X U_i = X' U_i (compare X_fa and X'fa, abs values to deal with different frames)
    assert torch.allclose(
        torch.abs(b[0].pos @ b[0].fa_rot[0].squeeze(0)),
        torch.abs(rotated_b[0].pos @ rotated_b[0].fa_rot[0].squeeze(0)),
        atol=10e-03,
    )
    # Check: U_i' = R U_i

    # SignNet model
    class SignNet(torch.nn.Module):
        def __init__(self, in_channels=3, hidden_channels=12, out_channels=3):
            super(SignNet, self).__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, out_channels),
            )
            torch.nn.init.xavier_uniform_(self.mlp[0].weight)
            torch.nn.init.xavier_uniform_(self.mlp[2].weight)
            self.mlp2 = torch.nn.Linear(3 * out_channels, 3 * out_channels)

            torch.nn.init.xavier_uniform_(self.mlp2.weight)

        def forward(self, x, second_mlp=False):
            if second_mlp:
                res = self.mlp(x) + self.mlp(-x)
                res = res.view(-1)  # flatten res
                res = self.mlp2(res)
                return res.view((3, -1)).T  # reshape as eigenvector column matrix
            return (self.mlp(x) + self.mlp(-x)).T

    signnet = SignNet()
    second_mlp = True

    for i in range(len(b.sid)):
        g = Batch.get_example(b, i)
        rotated_g = Batch.get_example(rotated_b, i)

        # Test: X_fa = R X_fa'
        torch.allclose(rotation_matrix @ rotated_g.fa_rot[0], g.fa_rot[0], atol=5e-01)

        # SignNet on eigenvector matrix U for g and rotated_g
        # Need SignNet(U_i) = U*, for every frame U_i
        # Eigenvectors are the columns of fa_rot. Need rows for SignNet MLPs
        eigen = signnet(g.fa_rot[0].squeeze(0).T, second_mlp)
        eigen_bis = signnet(g.fa_rot[1].squeeze(0).T, second_mlp)
        assert torch.allclose(eigen, eigen_bis, atol=1e-04)

        # Compare with rotated graph
        rot_eigen = signnet(rotated_g.fa_rot[0].squeeze(0).T, second_mlp)
        # Check U*' = R U*
        if torch.allclose(rot_eigen, eigen, atol=1e-4):
            print("U* is invariant to rotations")
        elif torch.allclose(rot_eigen, rotation_matrix @ eigen, atol=1e-4):
            print("U* is equivariant to rotations")
        else:
            print("U* is neither invariant nor equivariant")
        # Double-Check: X U* = X' U*'
        new_pos = g.pos @ eigen
        new_rotated_pos = rotated_g.pos @ rot_eigen
        if not torch.allclose(new_pos, new_rotated_pos, atol=1e-4):
            print("No equivariance: X U* != X' U*'")

        # Different eigenvalues matrix => want different U*
        m = g.fa_rot[0].squeeze(0).T + torch.randn(3, 3)
        e = signnet(m, second_mlp)
        if torch.allclose(e, eigen, atol=1e-4):
            print("Issue: distinct graph has same signnet eigenvectors")
        
        # Same but on real eigenvec matrix
        next_g = Batch.get_example(b, i+1)
        e = signnet(next_g.fa_rot[0].squeeze(0).T, second_mlp)
        if torch.allclose(e, eigen, atol=1e-4):
            print("Issue: distinct graph has same signnet eigenvectors")

        # Try with more complex network
        # Repalce False by True in signnet above
