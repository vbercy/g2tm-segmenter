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


import torch
from torch import nn
import torch.nn.functional as F

from segm.model.utils import padding, unpadding


class Segmenter(nn.Module):
    """ Segmenter model.
    """
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.token_reduction = False

    @torch.jit.ignore
    def no_weight_decay(self):
        """ Modules with no weight decay.
        """
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = (
            append_prefix_no_weight_decay("encoder.", self.encoder)
            .union(append_prefix_no_weight_decay("decoder.", self.decoder))
        )
        return nwd_params

    def forward(self, im):
        """ Forward function.
        """
        h_ori, w_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        h, w = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        if self.token_reduction:
            self.encoder.info["size"] = (
                self.encoder.info["size"][:, num_extra_tokens:]
            )
            if self.encoder.info["mask"] is not None:
                self.encoder.info["mask"] = (
                    self.encoder.info["mask"][:, num_extra_tokens:]
                )

        masks = self.decoder(x, (h, w), self.token_reduction)

        masks = F.interpolate(masks, size=(h, w), mode="bilinear")
        masks = unpadding(masks, (h_ori, w_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        """ Get attention map from a specified layer of the encoder.
        """
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        """ Get attention map from a specified layer of the decoder.
        """
        x, _ = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        if self.token_reduction:
            self.encoder.info["size"] = (
                self.encoder.info["size"][:, num_extra_tokens:]
            )
            if self.info["mask"] is not None:
                self.info["mask"] = self.info["mask"][:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id,
                                              self.token_reduction)
