import functools


@functools.total_ordering
class LabelPair(object):
    """
    Class to hold a pair of labels.

    Attributes:
        ref (Label): Reference label.
        hyp (Label): Hypothesis label.
    """

    def __init__(self, ref, hyp):
        self.ref = ref
        self.hyp = hyp

    def __lt__(self, other):
        label_a = self.ref
        label_b = other.ref

        if label_a is None:
            label_a = self.hyp

        if label_b is None:
            label_b = other.hyp

        if label_a is None:
            return True

        if label_b is None:
            return False

        return label_a < label_b

    def __eq__(self, other):
        if self.ref is None:
            return other.ref is None

        if self.hyp is None:
            return other.hyp is None

        if other.ref is None:
            return self.ref is None

        if other.hyp is None:
            return self.hyp is None

        return self.ref == other.ref and self.hyp == other.hyp

    def __repr__(self) -> str:
        return 'LabelPair({}, {})'.format(self.ref, self.hyp)

    def padded_ref_value(self):
        """
        Return the reference value as string padded to the longer value of ref and hyp.
        """
        length = self.max_length()
        value = '-' if self.ref is None else self.ref.value

        return value.rjust(length)

    def padded_hyp_value(self):
        """
        Return the hypothesis value as string padded to the longer value of ref and hyp.
        """
        length = self.max_length()
        value = '-' if self.hyp is None else self.hyp.value

        return value.rjust(length)

    def max_length(self):
        """
        Return the length of the longer value from ref and hyp.
        """
        length = 0

        if self.ref is not None:
            length = max(length, len(self.ref.value))
        else:
            length = max(length, len('None'))

        if self.hyp is not None:
            length = max(length, len(self.hyp.value))
        else:
            length = max(length, len('None'))

        return length


class Segment:
    """
    A class representing a segment within an alignment.

    Arguments:
        start (float): The start time in seconds.
        end (float): The end time in seconds.

    Attributes:
        ref (Label, list): List of or single reference label in the segment.
        hyp (Label, list): List of or single hypothesis label in the segment.
    """

    def __init__(self, start, end, ref=None, hyp=None):
        self.start = start
        self.end = end

        self.ref = ref
        self.hyp = hyp

    @property
    def duration(self):
        return self.end - self.start

    def __lt__(self, other):
        if self.start != other.start:
            return self.start < other.start

        if self.end != other.end:
            return self.end < other.end

        if self.ref != other.ref:
            if self.ref is None:
                return True
            elif other.ref is None:
                return False
            else:
                return self.ref < other.ref

        if self.hyp != other.hyp:
            if self.hyp is None:
                return True
            elif other.hyp is None:
                return False
            else:
                return self.hyp < other.hyp

        return False

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.ref == other.ref and self.hyp == other.hyp

    def __repr__(self):
        return '{} - {} REF: {} HYP: {}'.format(self.start, self.end, self.ref, self.hyp)
