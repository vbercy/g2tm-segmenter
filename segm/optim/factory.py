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


from timm import scheduler
from timm import optim

from segm.optim.scheduler import PolynomialLR


def create_scheduler(opt_args, optimizer):
    """ Instanciate scheduler.
    """
    if opt_args.sched == "polynomial":
        lr_scheduler = PolynomialLR(
            optimizer=optimizer,
            step_size=opt_args.poly_step_size,
            max_epoch=opt_args.epochs,
            power=opt_args.poly_power,
            min_lr=opt_args.min_lr,
            warmup_epochs=opt_args.warmup_epochs,
            start_factor=opt_args.start_factor,
        )
    else:
        lr_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return lr_scheduler


def create_optimizer(opt_args, model):
    """ Instanciate optimizer.
    """
    return optim.create_optimizer(opt_args, model)
