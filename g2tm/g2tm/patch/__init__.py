"""
Patching function for Segmenter models by inserting G2TM within one of its
encoder's Transformer blocks.

Example: Inserting G2TM@2[0.88] within a Segmenter model
    >>> from segm.model.factory import create_segmenter
    >>> from g2tm.graph import graph_segmenter_patch
    >>> model = create_segmenter({...})
    >>> selected_layer = 2
    >>> threshold = 0.88
    >>> graph_segmenter_patch(model, selected_layer, threshold)
"""

from .graph_segmenter_patch import apply_patch as graph_segmenter_patch

__all__ = ["graph_segmenter_patch"]
