import abc


class EventAligner(abc.ABC):
    """
    Abstract class for aligner classes that return a mapping between labels (events).

    An alignment is a mapping between labels from the ground truth (ref) and the system output (hyp).
    If there is no matching label in the system output for a label in the ground truth,
    it has to be aligned to ``None`` and vice versa. A single label can be aligned to multiple other labels.
    """

    @abc.abstractmethod
    def align(self, ref, hyp):
        """
        Return an alignment between the labels of the two label-lists.

        Args:
            ref (audiomate.corpus.assets.LabelList): The label-list containing labels of the ground truth.
            hyp (audiomate.corpus.assets.LabelList): The label-list containing labels of the system output.

        Returns:
            list: A list of :class:`evalmate.alignment.LabelPair`. Every pair contains one label from
            the ground truth and one from the system output, that are aligned. One of them also can be ``None``.
        """
        raise NotImplementedError()


class SegmentAligner(abc.ABC):
    """
    Abstract class for aligner classes that align labels in segments.

    An alignment is represented as a list of Segments with start/end-time and the labels
    from the ground truth and the system output, that are within this segment.
    """

    @abc.abstractmethod
    def align(self, ref, hyp):
        """
        Return an alignment of segments.

        Args:
            ref (audiomate.corpus.assets.LabelList): The label-list containing labels of the ground truth.
            hyp (audiomate.corpus.assets.LabelList): The label-list containing labels of the system output.

        Returns:
            list: A list of :class:`evalmate.utils.structure.Segment`. Every segment has start/end-time and
            two lists of labels that are contained in the segment
            (one for the ground truth and one for the system output).
        """
        raise NotImplementedError()
