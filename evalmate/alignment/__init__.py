"""
This module contains functionality for aligning two sequences of labels.

.. currentmodule:: evalmate.alignment

One-to-One
----------
Align every label from the hypothesis to a label from the reference if possible.

.. autoclass:: BipartiteMatchingOneToOneAligner
   :members:

Segment-Based
-------------

.. autoclass:: SegmentAligner
   :members:
"""

from .one_to_one import OneToOneAligner  # noqa: F401
from .one_to_one import BipartiteMatchingOneToOneAligner  # noqa: F401

from .segments import SegmentAligner  # noqa: F401
