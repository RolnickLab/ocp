"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import subprocess
from copy import deepcopy
import torch
import torch.distributed as dist


def setup(config):
    assert config["distributed"]
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    if node_list is not None:
        try:
            hostnames = subprocess.check_output(
                ["scontrol", "show", "hostnames", node_list]
            )
            config["init_method"] = "tcp://{host}:{port}".format(
                host=hostnames.split()[0].decode("utf-8"),
                port=config["distributed_port"],
            )
            nnodes = int(os.environ.get("SLURM_NNODES"))
            ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
            if ntasks_per_node is not None:
                ntasks_per_node = int(ntasks_per_node)
            else:
                ntasks = int(os.environ.get("SLURM_NTASKS"))
                nnodes = int(os.environ.get("SLURM_NNODES"))
                assert ntasks % nnodes == 0
                ntasks_per_node = int(ntasks / nnodes)
            if ntasks_per_node == 1:
                assert config["world_size"] % nnodes == 0
                gpus_per_node = config["world_size"] // nnodes
                node_id = int(os.environ.get("SLURM_NODEID"))
                config["rank"] = node_id * gpus_per_node
                config["local_rank"] = 0
            else:
                assert ntasks_per_node == config["world_size"] // nnodes
                config["rank"] = int(os.environ.get("SLURM_PROCID"))
                config["local_rank"] = int(os.environ.get("SLURM_LOCALID"))

            logging.info(
                f"Init: {config['init_method']}, {config['world_size']},"
                + f" {config['rank']}"
            )
            dist.init_process_group(
                backend=config["distributed_backend"],
                init_method=config["init_method"],
                world_size=config["world_size"],
                rank=config["rank"],
            )
        except subprocess.CalledProcessError as e:  # scontrol failed
            raise e
        except FileNotFoundError:  # Slurm is not installed
            pass


def cleanup():
    dist.destroy_process_group()


def initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if initialized() else 0


def get_world_size():
    return dist.get_world_size() if initialized() else 1


def is_master():
    return get_rank() == 0


def synchronize():
    if get_world_size() == 1:
        return
    dist.barrier()


def broadcast(tensor, src, group=dist.group.WORLD, async_op=False):
    if get_world_size() == 1:
        return
    dist.broadcast(tensor, src, group, async_op)


def all_reduce(data, group=dist.group.WORLD, average=False, device=None):
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    dist.all_reduce(tensor, group=group)
    if average:
        tensor /= get_world_size()
    if not isinstance(data, torch.Tensor):
        result = tensor.cpu().numpy() if tensor.numel() > 1 else tensor.item()
    else:
        result = tensor
    return result


def all_gather(data, group=dist.group.WORLD, device=None):
    if get_world_size() == 1:
        return data
    world_size = get_world_size()
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    tensor_list = [
        tensor.new_zeros(tensor.shape)
        for _ in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    new_tensor_list = [
        t.to(data.dtype if isinstance(data, torch.Tensor) else torch.int64)
        for t in tensor_list
    ]
    if not isinstance(data, torch.Tensor):
        result = [tensor.cpu().numpy() for tensor in new_tensor_list]
    else:
        result = new_tensor_list

    PRINTS = True
    if PRINTS:
        print("\n<>all_gather<>")
        print("tensor", tensor)
        print("tensor.dtype", tensor.dtype)
        print("tensor.float()", tensor.float())
        print(locals())
        print("</>all_gather</>\n")

    return result

    if isinstance(result, torch.Tensor):
        if (result > 1e6).any():
            print("<>all_gather<>")
            print(locals())
            print("</>all_gather</>")
    elif isinstance(result, list):
        if isinstance(result[0], torch.Tensor):
            if any((x > 1e6).any() for x in result):
                print("<>all_gather<>")
                print(locals())
                print("</>all_gather</>")
        elif any(x > 1e6 for x in result):
            print("<>all_gather<>")
            print(locals())
            print("</>all_gather</>")

    return result
