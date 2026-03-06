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


from itertools import cycle

import numpy as np


class Lines:
    """ Plot line.
    """
    def __init__(self, resolution=20, smooth=None):
        self.colors = cycle(
            [
                "#377eb8",
                "#e41a1c",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
                "#ffff33",
                "#a65628",
                "#f781bf",
            ]
        )
        self.markers = cycle("os^Dp>d<")
        self.legend = {"fontsize": 'medium', "labelspacing": 0, "numpoints": 1}
        self._resolution = resolution
        self._smooth_weight = smooth

    def __call__(self, ax, domains, lines, labels):
        assert len(domains) == len(lines) == len(labels)
        clrs = []
        for index, (label, color) in enumerate(zip(labels, self.colors)):
            domain, line = domains[index], lines[index]
            line = self.smooth(line, self._smooth_weight)
            ax.plot(domain, line[:, 0], color=color, label=label)

            last_x, last_y = domain[-1], line[-1, 0]
            ax.scatter(last_x, last_y, color=color, marker="x")
            ax.annotate(
                f"{last_y:.2f}",
                xy=(last_x, last_y),
                xytext=(last_x, last_y + 0.1),
            )
            clrs.append(color)

        self._plot_legend(ax, lines, labels)
        return clrs

    def _plot_legend(self, ax, lines, labels):
        # scores = {label: -np.nanmedian(line) for label,
        #           line in zip(labels, lines)}
        handles, labels = ax.get_legend_handles_labels()
        # handles, labels = zip(*sorted(
        #     zip(handles, labels), key=lambda x: scores[x[1]]))
        legend = ax.legend(handles, labels, **self.legend)
        legend.get_frame().set_edgecolor("white")
        for line in legend.get_lines():
            line.set_alpha(1)

    def smooth(self, scalars, weight):
        """
        weight in [0, 1]
        exponential moving average, same as tensorboard
        """
        assert weight >= 0 and weight <= 1
        last = scalars[0]
        smoothed = np.asarray(scalars)
        for i, point in enumerate(scalars):
            last = last * weight + (1 - weight) * point
            smoothed[i] = last

        return smoothed
