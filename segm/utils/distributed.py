# Copyright © 2025 Commissariat à l'Energie Atomique et aux Energies Alternatives (CEA)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications based on code from Robin Strudel et al. (Segmenter)

# MIT License

# Copyright (c) 2021 Robin Strudel
# Copyright (c) INRIA

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from pathlib import Path
import builtins as __builtin__

import torch
import torch.distributed as dist

import segm.utils.torch as ptu


def init_process(backend="nccl"):
    """ Init CUDA process group.
    """
    print(f"Starting process with rank {ptu.dist_rank} ({ptu.master_addr}:"
          f"{ptu.master_port})...", flush=True)

    torch.cuda.set_device(ptu.dist_rank)

    dist.init_process_group(
        backend,
        rank=ptu.dist_rank,
        world_size=ptu.world_size,
    )
    print(f"Process {ptu.dist_rank} is connected.", flush=True)
    dist.barrier()

    silence_print(ptu.dist_rank == 0)
    print("All processes are connected.", flush=True)


def silence_print(is_master):
    """ This function disables printing when not in master process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def sync_model(sync_dir, model):
    """ Sync model accross ranks.
    https://github.com/ylabbe/cosypose/blob/master/cosypose/utils/distributed.py
    """
    sync_path = Path(sync_dir).resolve() / "sync_model.pkl"
    if ptu.dist_rank == 0 and ptu.world_size > 1:
        torch.save(model.state_dict(), sync_path)
    dist.barrier()
    if ptu.dist_rank > 0:
        model.load_state_dict(torch.load(sync_path))
    dist.barrier()
    if ptu.dist_rank == 0 and ptu.world_size > 1:
        sync_path.unlink()
    return model


def barrier():
    """ Redefining torch.distributed.barrier() function.
    """
    dist.barrier()


def destroy_process():
    """ Redefining torch.distributed.destroy_process_group() function.
    """
    dist.destroy_process_group()
