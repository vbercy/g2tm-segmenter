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

# Adapted from 2020 Ross Wightman (https://github.com/rwightman/pytorch-image-models)


from typing import Tuple

import torch

from segm.model.blocks import Attention, Block


TwoTensors = Tuple[torch.Tensor, torch.Tensor]
ThreeTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class MaskedAttention(Attention):
    """Multi-head self-attention layer with optional token masking.

    This module applies a self-attention operation to a token feature sequence
    and optionally ignores tokens using a binary mask.

    Args:
        dim (int): Embedding dimension of the tokens.
        heads (int): Number of attention heads.
        dropout (float): dropout probability applied to attention weights.

    Attributes:
        heads (int): Number of attention heads.
        scale (float): Scaling factor applied to attention scores.
    """

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> TwoTensors:
        """Multi-head self-attention implem. with optional token masking.

        This method applies masked self-attention to a token feature sequence.

        Args:
            x (torch.Tensor): Token features (b, n, c).
            mask (torch.Tensor): Binary mask (b, n), where False indicates
                that the token has been fused token and has not been assigned
                the mean value.
            size (torch.Tensor): Size of each token (i.e.: number of original
                patches it represents) (b, n).

        Returns:
            x (torch.Tensor): Modified token features
            attn (torch.Tensor): Corresponding attention map
        """
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class ProportionalAttention(Attention):
    """Multi-head proportional self-attention layer with optional token
    masking.

    This module applies a proportional self-attention operation (from ToMe) to
    a token feature sequence. Optionally, it adds a log term in the attention
    computation depending on the token sizes. The same term is used for masking
    tokens instead of using a binary mask.

    Args:
        dim (int): Embedding dimension of the tokens.
        heads (int): Number of attention heads.
        dropout (float): dropout probability applied to attention weights.

    Attributes:
        heads (int): Number of attention heads.
        scale (float): Scaling factor applied to attention scores.
    """

    def forward(self, x: torch.Tensor,
                size: torch.Tensor = None) -> TwoTensors:
        """Multi-head proportional self-attention implementation with optional
        token masking.

        This method applies proportional masked self-attention to a token
        feature sequence.

        Args:
            x (torch.Tensor): Token features (b, n, c).
            mask (torch.Tensor): Binary mask (b, n), where False indicates
                that the token has been fused token and has not been assigned
                the mean value.
            size (torch.Tensor): Size of each token (i.e.: number of original
                patches it represents) (b, n).

        Returns:
            x (torch.Tensor): Modified token features
            attn (torch.Tensor): Corresponding attention map
        """
        b, n, c = x.shape

        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Using log(0)=-inf for padded to 0 tokens to apply both propor and
        # masked attn
        if size is not None:
            attn = attn + size.log()[:, None, None, :]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class InverseProportionalAttention(Attention):
    """Multi-head inverse proportional self-attention layer with optional
    token masking.

    This module applies an inverse proportional self-attention operation (from
    G2TM) to a token feature sequence. Optionally, it substracts a log term in
    the attention computation depending on the token sizes. The same term is
    used for masking tokens instead of using a binary mask.

    Args:
        dim (int): Embedding dimension of the tokens.
        heads (int): Number of attention heads.
        dropout (float): dropout probability applied to attention weights.

    Attributes:
        heads (int): Number of attention heads.
        scale (float): Scaling factor applied to attention scores.
    """

    def forward(self, x: torch.Tensor,
                size: torch.Tensor = None) -> TwoTensors:
        """Multi-head inverse proportional self-attention implementation with
        optional token masking.

        This method applies inverse proportional masked self-attention to a
        token feature sequence.

        Args:
            x (torch.Tensor): Token features (b, n, c).
            mask (torch.Tensor): Binary mask (b, n), where False indicates
                that the token has been fused token and has not been assigned
                the mean value.
            size (torch.Tensor): Size of each token (i.e.: number of original
                patches it represents) (b, n).

        Returns:
            x (torch.Tensor): Modified token features
            attn (torch.Tensor): Corresponding attention map
        """
        b, n, c = x.shape

        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply inverted proportional attention and fixing attention of all
        # padded to 0 tokens to -inf to apply masked attn at the same time
        if size is not None:
            log_size = size.log().masked_fill(size == 0, float('inf'))
            attn.sub_(log_size[:, None, None, :])

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MaskedBlock(Block):
    """Multi-head masked self-attention block.

    This module coresponds to the entire masked self-attention block. It
    gathers the masked attention, the normalization and the MLP layers.

    Args:
        dim (int): Embedding dimension of the tokens.
        heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of the MLP layer.
        dropout (float): dropout probability applied to attention weights.
        drop_path (float): Probability of dropping the residual path.

    Attributes:
        heads (int): Number of attention heads.
        scale (float): Scaling factor applied to attention scores.
    """

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__(dim, heads, mlp_dim, dropout, drop_path)
        self.attn = MaskedAttention(dim, heads, dropout)

    def forward(self, x, return_attention=False):
        """Multi-head masked self-attention block implementation.

        This method applies all the layers of the masked self-attention block
        to a token feature sequence. It gets the binary mask from a
        dictionnary initialized by the G2TM patch.

        Args:
            x (torch.Tensor): Token features (B, N, C).
            return_attention (bool): Wether the features (False) or the
                attention map (True) should be returned.

        Returns:
            x (torch.Tensor): Modified token features
            OR attn (torch.Tensor): Corresponding attention map
        """
        mask = self.info["mask"]
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ProportionalBlock(Block):
    """Multi-head proportional self-attention block.

    This module coresponds to the entire proportional self-attention block. It
    gathers the proportional attention, the normalization and the MLP layers.

    Args:
        dim (int): Embedding dimension of the tokens.
        heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of the MLP layer.
        dropout (float): dropout probability applied to attention weights.
        drop_path (float): Probability of dropping the residual path.

    Attributes:
        heads (int): Number of attention heads.
        scale (float): Scaling factor applied to attention scores.
    """

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__(dim, heads, mlp_dim, dropout, drop_path)
        self.attn = ProportionalAttention(dim, heads, dropout)

    def forward(self, x, return_attention=False):
        """Multi-head proportional self-attention block implementation.

        This method applies all the layers of the proportional self-attention
        block to a token feature sequence. It gets the size vectors from a
        dictionnary initialized by the G2TM patch.

        Args:
            x (torch.Tensor): Token features (B, N, C).
            return_attention (bool): Wether the features (False) or the
                attention map (True) should be returned.

        Returns:
            x (torch.Tensor): Modified token features
            OR attn (torch.Tensor): Corresponding attention map
        """
        attn_size = self.info["size"]
        y, attn = self.attn(self.norm1(x), size=attn_size)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class InverseProportionalBlock(Block):
    """Multi-head inverse proportional self-attention block.

    This module coresponds to the entire inverse proportional self-attention
    block. It gathers the inverse proportional attention, the normalization
    and the MLP layers.

    Args:
        dim (int): Embedding dimension of the tokens.
        heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of the MLP layer.
        dropout (float): dropout probability applied to attention weights.
        drop_path (float): Probability of dropping the residual path.

    Attributes:
        heads (int): Number of attention heads.
        scale (float): Scaling factor applied to attention scores.
    """

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__(dim, heads, mlp_dim, dropout, drop_path)
        self.attn = InverseProportionalAttention(dim, heads, dropout)

    def forward(self, x, return_attention=False):
        """Multi-head inverse proportional self-attention block implementation.

        This method applies all the layers of the inverse proportional
        self-attention block to a token feature sequence. It gets the size
        vectors from a dictionnary initialized by the G2TM patch.

        Args:
            x (torch.Tensor): Token features (B, N, C).
            return_attention (bool): Wether the features (False) or the
                attention map (True) should be returned.

        Returns:
            x (torch.Tensor): Modified token features
            OR attn (torch.Tensor): Corresponding attention map
        """
        attn_size = self.info["size"]
        y, attn = self.attn(self.norm1(x), size=attn_size)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
