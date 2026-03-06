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


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class Loader(DataLoader):
    """ Dataloader class.
    """
    def __init__(self, dataset, batch_size, num_workers, distributed, split):
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
            )
        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

        self.base_dataset = self.dataset

    @property
    def unwrapped(self):
        """ Unwrap.
        """
        return self.base_dataset.unwrapped

    def set_epoch(self, epoch):
        """ Set sampler epoch.
        """
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    def get_diagnostics(self, logger):
        """ Get diagnostics from logger.
        """
        return self.base_dataset.get_diagnostics(logger)

    def get_snapshot(self):
        """ Get snapshot.
        """
        return self.base_dataset.get_snapshot()

    def end_epoch(self, epoch):
        """ Get end epoch.
        """
        return self.base_dataset.end_epoch(epoch)
