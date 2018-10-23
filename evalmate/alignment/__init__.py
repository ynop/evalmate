"""
This module contains functionality for aligning labels of a ground truth with the labels of a system output.

.. currentmodule:: evalmate.alignment


Base classes
------------
All aligners are based either on :class:`EventAligner` or :class:`SegmentAligner`.
The base classes are mainly distinguished by the type of the alignment they return.
While the :class:`EventAligner` returns a mapping between complete labels,
the :class:`SegmentAligner` returns segments, that can span over parts of labels.

.. autoclass:: EventAligner
   :members:

.. autoclass:: SegmentAligner
   :members:

Time-Based
-----------
Align labels based on some distance metric based on their start/endtimes.

.. autoclass:: BipartiteMatchingAligner
   :members:

.. autoclass:: FullMatchingAligner
   :members:

Sequence-Based
--------------
Align labels only considering the ordering of the sequence.

.. autoclass:: LevenshteinAligner
   :members:

Segment-Based
-------------
Align labels based on segments defined by start/end-time.

.. autoclass:: InvariantSegmentAligner
   :members:

Utils
-----

.. autoclass:: Segment
   :members:

.. autoclass:: LabelPair
   :members:
"""

from .aligner import EventAligner  # noqa: F401
from .aligner import SegmentAligner  # noqa: F401

from .time import BipartiteMatchingAligner  # noqa: F401
from .time import FullMatchingAligner  # noqa: F401

from .sequence import LevenshteinAligner  # noqa: F401

from .segment import InvariantSegmentAligner  # noqa: F401

from .utils import Segment  # noqa: F401
from .utils import LabelPair  # noqa: F401
