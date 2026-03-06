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


from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    """ Polynomial learning rate scheduler.
    """
    def __init__(
        self,
        optimizer,
        step_size,
        max_epoch,
        power,
        min_lr=1e-5,
        last_epoch=-1,
        warmup_epochs=0,
        start_factor=1.,
    ):
        self.step_size = step_size
        self.warmup_epochs = int(warmup_epochs)
        self.start_factor = start_factor
        self.max_epoch = int(max_epoch)
        self.power = power
        self.min_lr = min_lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

        if self.warmup_epochs > 0:
            print(self.start_factor)
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr'] * self.start_factor

    def linear_warmup(self, lr, initial_lr):
        """NOTE: initial_lr = learning rate wanted at the end of the warmup
        """
        coef = (1. - self.start_factor) / self.warmup_epochs
        return lr + coef * initial_lr

    def polynomial_decay(self, initial_lr):
        """ Learning rate polynomial decay.
        """
        epoch_cur = int(self.last_epoch)
        coef = (1. - (epoch_cur - self.warmup_epochs) /
                (self.max_epoch - self.warmup_epochs)) ** self.power
        return (initial_lr - self.min_lr) * coef + self.min_lr

    def get_lr(self):
        if (
            self.last_epoch == -1
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.max_epoch)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_epochs:
            return [self.linear_warmup(group['lr'], group['initial_lr'])
                    for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]

    def step_update(self, last_epoch):
        """ Update step.
        """
        self.step(last_epoch)
