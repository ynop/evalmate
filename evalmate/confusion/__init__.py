"""
This module contains classes for computing confusion statistics.

.. currentmodule:: evalmate.confusion

Confusion
---------

.. autoclass:: Confusion
   :members:

SegmentConfusion
----------------

.. autoclass:: SegmentConfusion
   :members:

EventConfusion
----------------

.. autoclass:: EventConfusion
   :members:

AggregatedConfusion
-------------------

.. autoclass:: AggregatedConfusion
   :members:

"""

from .confusion import Confusion  # noqa: F401

from .segment import SegmentConfusion  # noqa: F401
from .event import EventConfusion  # noqa: F401
from .aggregation import AggregatedConfusion  # noqa: F401


def create_from_segments(segments):
    """
    Create confusion from a list of segments.
    ``self.instances`` will contain the SegmentConfusion for every value occurring in the given segments.

    Arguments:
        segments (list): List of Segments.

    Returns:
        AggregatedConfusion: The confusion.
    """

    cnf = AggregatedConfusion()

    for segment in segments:
        if segment.ref is not None and segment.ref.value not in cnf.instances.keys():
            cnf.instances[segment.ref.value] = SegmentConfusion(segment.ref.value)

        if segment.hyp is not None and segment.hyp.value not in cnf.instances.keys():
            cnf.instances[segment.hyp.value] = SegmentConfusion(segment.hyp.value)

        if segment.ref is None and segment.hyp is None:
            print('Got segment with ref=None and hyp=None, ignoring it!')

        elif segment.ref is None:
            cnf.instances[segment.hyp.value].insertion_segments.append(segment)

        elif segment.hyp is None:
            cnf.instances[segment.ref.value].deletion_segments.append(segment)

        elif segment.ref.value == segment.hyp.value:
            cnf.instances[segment.ref.value].correct_segments.append(segment)

        else:
            cnf.instances[segment.ref.value].substitution_segments[segment.hyp.value].append(segment)
            cnf.instances[segment.hyp.value].substitution_out_segments[segment.ref.value].append(segment)

    return cnf


def create_from_label_pairs(pairs):
    """
    Create confusion from a list of aligned labels.

    Arguments:
        pairs (list): List of LabelPair

    Returns:
        AggregatedConfusion: Confusion
    """

    cnf = AggregatedConfusion()

    for pair in pairs:
        if pair.ref is not None and pair.ref.value not in cnf.instances.keys():
            cnf.instances[pair.ref.value] = EventConfusion(pair.ref.value)

        if pair.hyp is not None and pair.hyp.value not in cnf.instances.keys():
            cnf.instances[pair.hyp.value] = EventConfusion(pair.hyp.value)

        if pair.ref is None and pair.hyp is None:
            print('Got label pair with ref=None and hyp=None, ignoring it!')

        elif pair.ref is None:
            cnf.instances[pair.hyp.value].insertion_pairs.append(pair)

        elif pair.hyp is None:
            cnf.instances[pair.ref.value].deletion_pairs.append(pair)

        elif pair.ref.value == pair.hyp.value:
            cnf.instances[pair.ref.value].correct_pairs.append(pair)

        else:
            cnf.instances[pair.ref.value].substitution_pairs[pair.hyp.value].append(pair)
            cnf.instances[pair.hyp.value].substitution_out_pairs[pair.ref.value].append(pair)

    return cnf
