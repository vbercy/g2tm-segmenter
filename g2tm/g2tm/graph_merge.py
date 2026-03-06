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

import math
from os import environ
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
import networkx as nx

environ["NETWORKX_AUTOMATIC_BACKENDS"] = "cugraph"
environ["NX_CUGRAPH_AUTOCONFIG"] = "True"

MayBeTensor = Union[torch.Tensor, None]
FourTensors = Tuple[torch.Tensor, torch.Tensor,
                    torch.Tensor, MayBeTensor]
TripleIntList = List[List[List[int]]]
DoubleIntList = List[List[int]]


def get_mergeable_idxs(x_feat: torch.Tensor, threshold: float,
                       base_grid_h: int, base_grid_w: int,
                       b: int, n: int,
                       device: torch.device) -> TripleIntList:
    """Determine the indices of the groups of tokens to merge.

    This function computes the cosine similarity between all tokens and their
    respective 4-neighbors. Two tokens are neighbors if and only if the patches
    they represent in the image are neighbors. The similarity scores are then
    thresholded. From the remaining edges between tokens, we create a NetworkX
    graph and apply a graph search algorithm (Breadth First Search) to group
    tokens to merge with each other into lists of indices.

    Args :
        x_feat (torch.Tensor): Token feature sequence.
        threshold (float): Threshold parameter for G2TM.
        base_grid_h (int): Image height in number of patches.
        base_grid_w (int): Image width in number of patches.
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        device (torch.device): Device where tensors are processed.

    Returns:
        connected_components (TripleIntList): Groups of indices corresponding
        to tokens to merge in each batch.
    """
    # === COSINE SIMILARITY ===

    # Determine indices that have a neighbor on its right and below it.
    # Here, we need to consider the tokens as they were placed like
    # the patches in the image.
    has_right_idxs = ((torch.arange(base_grid_h, device=device)[:, None]  # pylint: disable=E1101
                      * base_grid_w)
                      + torch.arange(base_grid_w - 1, device=device)).ravel()  # pylint: disable=E1101
    has_bottom_idxs = torch.arange(n - base_grid_w, device=device)  # pylint: disable=E1101

    # Compute cosine similarities of tokens with their right and bottom
    # neighbors
    right_sims = F.cosine_similarity(  # pylint: disable=E1102
        x_feat[:, has_right_idxs + 1, :], x_feat[:, has_right_idxs, :], dim=-1
    )
    bottom_sims = F.cosine_similarity(  # pylint: disable=E1102
        x_feat[:, has_bottom_idxs + base_grid_w, :],
        x_feat[:, has_bottom_idxs, :], dim=-1
    )

    # === CONNECTED COMPONENTS === #

    # Connection masks based on the similarity with the right and bottom
    # neighbors (if exist)
    right_mask = torch.zeros((b, n), dtype=bool, device=device)  # pylint: disable=E1101
    bottom_mask = torch.zeros((b, n), dtype=bool, device=device)  # pylint: disable=E1101
    node_mean = torch.tensor(threshold, device=device)  # pylint: disable=E1101
    right_mask[:, has_right_idxs] = right_sims > node_mean
    bottom_mask[:, has_bottom_idxs] = bottom_sims > node_mean

    # Get all the connected components for all the batches in a list
    # (i.e.: List[List[List, ...], ...])
    connected_components = []
    for bb in range(b):
        # Get indices of tokens having a valid connection with its right and
        # bottom neighbors
        right_connections = right_mask[bb].nonzero()  # .cpu() (a bit longer)
        bottom_connections = bottom_mask[bb].nonzero()  # .cpu() (a bit longer)
        # Concatenate the tensors to get the pairs of connected tokens
        src_nodes = torch.cat(  # pylint: disable=E1101
            (right_connections, bottom_connections), dim=0
        )
        dst_nodes = torch.cat(  # pylint: disable=E1101
            (right_connections + 1, bottom_connections + base_grid_w), dim=0
        )
        # Create NetworkX graph and find connected components
        graph = nx.Graph(torch.cat((src_nodes, dst_nodes), dim=-1).tolist())  # pylint: disable=E1101
        connected_components.append(
            list(map(list, nx.connected_components(graph)))
        )

    return connected_components


def padded_merge(x_feat: torch.Tensor,
                 connected_components: TripleIntList,
                 b: int, n: int, d: int,
                 device: torch.device) -> FourTensors:
    """Merge token features with each other according to indices in argument.

    This function computes the mean token feature per connected component. For
    each connected component, the token in the middle of the list will store
    the mean value, the others will be flagged in a binary mask for later
    masking in the attention computation. It also computes the source matrix,
    which track which token has been merged with another, and the size vector,
    which indicates the size of each token (i.e.: number of original patches
    it represents).

    Args:
        x_feat (torch.Tensor): Token feature sequence.
        connected_components (TripleIntList): Groups of indices corresponding.
        to tokens to merge in each batch.
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        d (int): Embedding dimension of the tokens.
        device (torch.device): Device where tensors are processed.

    Returns:
        x_feat (torch.Tensor): Reduced token feature sequence.
        source (torch.Tensor): Matrix tracking fusions between tokens.
        size (torch.Tensor): Vector storing token sizes.
        mask (torch.Tensor): Binary mask for reduced tokens.
    """

    source = torch.eye(n, dtype=int, device=device).expand(b, n, n).clone()  # pylint: disable=E1101
    mask = torch.ones(b, n, dtype=bool, device=device)  # pylint: disable=E1101

    for k, cc in enumerate(connected_components):
        if len(cc) > 0:
            # For each connected component we want the :
            # - middle index which will be assigned the mean value (mean_idxs)
            # - other indices to compute the mean value (to_reduce_idxs)
            # - index of the associated "mean_idxs" for each "to_reduce_idxs"
            mean_idxs = []
            to_reduce_idxs = []
            merge_idxs = []

            for i, fused_tokens in enumerate(cc):
                n_tokens = len(fused_tokens)
                mean_idxs.append(fused_tokens.pop(n_tokens//2))
                to_reduce_idxs += fused_tokens
                merge_idxs += [[i]] * (n_tokens - 1)

            mean_idxs = torch.tensor(mean_idxs, device=device)  # pylint: disable=E1101
            to_reduce_idxs = torch.tensor(to_reduce_idxs, device=device)  # pylint: disable=E1101
            merge_idxs = torch.tensor(merge_idxs, device=device)  # pylint: disable=E1101

            # Compute the mean tokens for each connected component and the
            # corresponding source vectors
            x_feat[k, mean_idxs] = (
                x_feat[k, mean_idxs].scatter_reduce(0,
                                                    merge_idxs.expand(-1, d),
                                                    x_feat[k, to_reduce_idxs],
                                                    reduce="sum")
            )
            source[k, mean_idxs] = (
                source[k, mean_idxs].scatter_reduce(0,
                                                    merge_idxs.expand(-1, n),
                                                    source[k, to_reduce_idxs],
                                                    reduce="amax")
            )

            # Fixing all fused tokens to 0 is not necessary as we use masked
            # attention
            mask[k, to_reduce_idxs] = False
    source *= mask.unsqueeze(-1)

    size = source.sum(-1)
    fused_mask = size > 1
    x_feat[fused_mask] = x_feat[fused_mask] / size[fused_mask].unsqueeze(1)

    return x_feat, source, size, mask


def sparse_merge(x_feat: torch.Tensor,
                 connected_components: DoubleIntList,
                 n: int, d: int, device: torch.device) -> FourTensors:
    """Merge token features with each other according to indices in argument.

    This function computes the mean token feature per connected component. For
    each connected component, a token in the list will store the mean value,
    while the others will be discarded. Thus, we do not have to pass a binary
    mask like in `padded_merge` (so None is returned). It also computes the
    source matrix, which track which token has been merged with another, and
    the size vector, which indicates the size of each token (i.e.: number of
    original patches it represents).

    WARNING: This function can only be applied when only 1 image per batch is
    processed.

    Args:
        x_feat (torch.Tensor): Token feature sequence.
        connected_components (DoubleIntList): Groups of indices corresponding
        to tokens to merge (only one batch in this case).
        n (int): Number of tokens in the sequence.
        d (int): Embedding dimension of the tokens.
        device (torch.device): Device where tensors are processed.

    Returns:
        x_feat (torch.Tensor): Reduced token feature sequence.
        source (torch.Tensor): Matrix tracking fusions between tokens.
        size (torch.Tensor): Vector storing token sizes.
        None: Instead of the binary mask.
    """

    source = torch.eye(n, dtype=int, device=device).expand(1, n, n).clone()  # pylint: disable=E1101
    mask = torch.ones(n, dtype=bool, device=device)  # pylint: disable=E1101
    size = torch.ones(1, n, dtype=int, device=device)  # pylint: disable=E1101

    if len(connected_components) > 0:
        # For each connected component we want the :
        # - middle index which will be assigned the mean value (mean_idxs)
        # - other indices to compute the mean value (to_reduce_idxs)
        # - index of the associated "mean_idxs" for each "to_reduce_idxs"
        mean_idxs = []
        to_reduce_idxs = []
        merge_idxs = []

        for i, fused_tokens in enumerate(connected_components):
            n_tokens = len(fused_tokens)
            mean_idxs.append(fused_tokens.pop(n_tokens//2))
            to_reduce_idxs += fused_tokens
            merge_idxs += [[i]] * (n_tokens - 1)

        mean_idxs = torch.tensor(mean_idxs, device=device)  # pylint: disable=E1101
        to_reduce_idxs = torch.tensor(to_reduce_idxs, device=device)  # pylint: disable=E1101
        merge_idxs = torch.tensor(merge_idxs, device=device)  # pylint: disable=E1101

        # Compute the mean tokens for each connected component and the
        # corresponding source vectors
        x_feat[0, mean_idxs] = (
            x_feat[0, mean_idxs].scatter_reduce(0, merge_idxs.expand(-1, d),
                                                x_feat[0, to_reduce_idxs],
                                                reduce="sum")
        )
        source[0, mean_idxs] = (
            source[0, mean_idxs].scatter_reduce(0, merge_idxs.expand(-1, n),
                                                source[0, to_reduce_idxs],
                                                reduce="amax")
        )

        # Pop all fused tokens except the ones we assign the means
        mask[to_reduce_idxs] = False
        x_feat = x_feat[:, mask]
        source = source[:, mask]

        size = source.sum(-1)
        x_feat = x_feat / size.unsqueeze(-1)

    return x_feat, source, size, None


def g2tm_merge(feat: torch.Tensor, threshold: float,
               is_encoder: bool = True,
               distill_token: bool = False) -> FourTensors:
    """Applies the entire G2TM processing to a token feature sequence.

    This function determines the groups of token indices to merge with each
    other and then computes the merged token associated to each of these
    groups. It separates the cases where the batch size is 1 or any other
    value to process the token sequence accordingly.

    Args:
        feat (torch.Tensor): Token feature sequence.
        threshold (float): Threshold parameter for G2TM.
        is_encoder (bool): Whether the G2TM have been inserted in a ViT
                           encoder.
        distill_token (bool): Whether the sequence contains a distillation
                              token.

    Returns:
        x_feat (torch.Tensor): Reduced token feature sequence.
        source (torch.Tensor): Matrix tracking fusions between tokens.
        size (torch.Tensor): Vector storing token sizes.
        mask (torch.Tensor|None): Binary mask for reduced tokens.
    """

    # G2TM only working with ViT encoder having 1 [CLS] token at the beginning
    # of the sequence
    protected = 0
    if is_encoder:
        protected += 1
    if distill_token:
        protected += 1

    x_protected, x_feat = feat[:, :protected, :], feat[:, protected:, :]
    b, n, d = x_feat.size()
    device = feat.device

    base_grid_h = int(math.sqrt(n))
    base_grid_w = base_grid_h
    assert base_grid_h * base_grid_w == n

    # Get the indices of tokens to merge
    with torch.no_grad():
        connected_components = get_mergeable_idxs(x_feat, threshold,
                                                  base_grid_h, base_grid_w,
                                                  b, n, device)

    # Merge tokens
    if b == 1:
        # Reduced tokens are popped
        x_feat, source, size, mask = (
            sparse_merge(x_feat, connected_components[0], n, d, device)
        )
    else:
        # Reduced tokens are filled with zeros
        x_feat, source, size, mask = (
            padded_merge(x_feat, connected_components, b, n, d, device)
        )
        mask = torch.cat(  # pylint: disable=E1101
            (torch.ones(b, protected, dtype=bool, device=device), mask), 1  # pylint: disable=E1101
        )

    feat = torch.cat((x_protected, x_feat), dim=1)  # pylint: disable=E1101
    size = torch.cat(  # pylint: disable=E1101
        (torch.ones(b, protected, dtype=int, device=device), size), 1  # pylint: disable=E1101
    )

    return feat, source, size, mask
