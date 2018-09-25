from evalmate import confusion
from evalmate.alignment import segments

from . import base


class ClassificationEvaluation(base.Evaluation):
    """
    Result of an evaluation of a keyword spotting task.

    Arguments:
        aligned_segments (list): List of :py:class:`evalmate.utils.structure.Segment`.

    Attributes:
        ref_outcome (Outcome): The outcome of the ground-truth/reference.
        hyp_outcome (Outcome): The outcome of the system-output/hypothesis.
        confusion (AggregatedConfusion): Confusion result
    """

    def __init__(self, ref_outcome, hyp_outcome, aligned_segments):
        super(ClassificationEvaluation, self).__init__(ref_outcome, hyp_outcome)

        self.aligned_segments = aligned_segments
        self.confusion = confusion.create_from_segments(self.aligned_segments)

    @property
    def default_template(self):
        return 'classification'

    @property
    def template_data(self):
        return {'confusion': self.confusion,
                'ref_outcome': self.ref_outcome,
                'hyp_outcome': self.hyp_outcome}


class ClassificationEvaluator(base.Evaluator):
    """
    Evaluation of a sequence classification task.
    In a sequence classification task every part of a sequence has to be assigned one out of some predefined classes.

    For example imagine a recording of a radio broadcast and
    the system has to classify every part of the signal either into music or speech.
    """

    def __init__(self):
        self.aligner = segments.SegmentAligner()

    @classmethod
    def default_label_list_idx(cls):
        return 'domain'

    def do_evaluate(self, ref, hyp):
        all = []

        for key, ll_ref in ref.label_lists.items():
            ll_hyp = hyp.label_lists[key]

            aligned_segments = self.aligner.align(ll_ref, ll_hyp)
            aligned_segments = ClassificationEvaluator.flatten_overlapping_labels(aligned_segments)

            all.extend(aligned_segments)

        return ClassificationEvaluation(ref, hyp, all)

    @staticmethod
    def flatten_overlapping_labels(aligned_segments):
        """
        Check all segments for overlapping labels.
        Overlapping means there are multiple reference or multiple hypothesis labels in a segment.

        Arguments:
            aligned_segments (List): List of segments.

        Returns:
            list: List of segments where ref and hyp is a single label.

        Raises:
            ValueError: A segment contains overlapping labels.
        """
        for segment in aligned_segments:
            if len(segment.ref) > 1:
                raise ValueError('Overlapping labels in reference.')

            if len(segment.hyp) > 1:
                raise ValueError('Overlapping labels in hypothesis.')

            if len(segment.ref) > 0:
                segment.ref = segment.ref[0]
            else:
                segment.ref = None

            if len(segment.hyp) > 0:
                segment.hyp = segment.hyp[0]
            else:
                segment.hyp = None

        return aligned_segments
