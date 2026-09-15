"""
G2TM - Graph-Guided Token Merging

This package provide tools for inserting and running G2TM inside a Segmenter
model.
It includes PyTorch modules, utilities and visualization functions.
"""

from . import graph_merge
from .patch import graph_segmenter_patch
from .vis import (
    make_visualization,
    make_overlayed_visualization,
    make_seg_visualization,
    add_grid,
)
from .version import __version__, version_info

__all__ = [
    "graph_merge",
    "graph_segmenter_patch",
    "make_visualization",
    "make_overlayed_visualization",
    "make_seg_visualization",
    "add_grid",
    "__version__",
    "version_info",
]
