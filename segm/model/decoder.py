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


from einops import rearrange
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from segm.model.blocks import Block
from segm.model.utils import init_weights


class DecoderLinear(nn.Module):
    """ Decoder = linear layer.
    """
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        """ Modules with no weight decay.
        """
        return set()

    def forward(self, x, im_size, token_reduction):
        """ Forward function.
        """
        h, _ = im_size
        gs = h // self.patch_size
        b, n, c = x.shape
        x = self.head(x)

        if token_reduction:
            if self.info["mask"] is None:
                idxs = self.info["source"].argmax(dim=1)
                x_ = torch.ones(1, n, c, device=x.device)  # pylint: disable=E1101
                x_[0, :, :] = x[0, idxs[0]]
            else:
                x_ = torch.ones(b, n, c, device=x.device)  # pylint: disable=E1101
                for batch in range(0, b):
                    idxs = self.info["source"][batch].argmax(dim=0)
                    x_[batch, :, :] = x[batch, idxs]
        else:
            x_ = x

        x_ = rearrange(x_, "b (h w) c -> b c h w", h=gs)

        return x_


class MaskTransformer(nn.Module):
    """ Decoder = 2-layer Mask Transformer.
    """
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, n_layers)  # pylint: disable=E1101
        ]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i])
             for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))  # pylint: disable=E1101
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(
            self.scale * torch.randn(d_model, d_model)  # pylint: disable=E1101
        )
        self.proj_classes = nn.Parameter(
            self.scale * torch.randn(d_model, d_model)  # pylint: disable=E1101
        )

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        """ Modules with no weight decay.
        """
        return {"cls_emb"}

    def forward(self, x, im_size, token_reduction):
        """ Forward function.
        """
        h, w = im_size
        gh = h // self.patch_size
        gw = w // self.patch_size
        b = x.size(0)

        x = self.proj_dec(x)

        cls_emb = self.cls_emb.expand(b, -1, -1)
        x = torch.cat((x, cls_emb), 1)  # pylint: disable=E1101

        if token_reduction:
            cls_extension = torch.ones(b, self.n_cls, dtype=int,  # pylint: disable=E1101
                                       device=x.device)
            self.info["size"] = torch.cat(  # pylint: disable=E1101
                (self.info["size"], cls_extension), 1
            )
            if self.info["mask"] is not None:
                self.info["mask"] = torch.cat(  # pylint: disable=E1101
                    (self.info["mask"], cls_extension.bool()), 1
                )

        for blk in self.blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, :-self.n_cls], x[:, -self.n_cls:]

        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)

        if token_reduction:
            masks_dim = masks.size(2)
            masks_ = torch.ones(b, gh*gw, masks_dim, device=x.device)  # pylint: disable=E1101
            for batch in range(0, b):
                idxs = self.info["source"][batch].argmax(dim=0)
                masks_[batch, :, :] = masks[batch, idxs]
        else:
            masks_ = masks
        masks_ = rearrange(masks_, "b (h w) n -> b n h w", h=int(gh))

        return masks_

    def get_attention_map(self, x, layer_id, token_reduction):
        """ Get attention map from a specified layer.
        """
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id}"
                f" < {self.n_layers}."
            )
        b = x.size(0)

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(b, -1, -1)
        x = torch.cat((x, cls_emb), 1)  # pylint: disable=E1101

        if token_reduction:
            cls_extension = torch.ones(b, self.n_cls, dtype=int,  # pylint: disable=E1101
                                       device=x.device)
            self.info["size"] = torch.cat(  # pylint: disable=E1101
                (self.info["size"], cls_extension), 1
            )
            if self.info["mask"] is not None:
                self.info["mask"] = torch.cat(  # pylint: disable=E1101
                    (self.info["mask"], cls_extension.bool()), 1
                )

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
