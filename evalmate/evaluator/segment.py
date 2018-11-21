from evalmate import confusion
from evalmate import alignment

from . import evaluator


class SegmentEvaluation(evaluator.Evaluation):
    """
    Result of an evaluation of a segment-based alignment.

    Arguments:
        utt_to_segments (dict): Dict of lists with :py:class:`evalmate.alignment.Segment`.
                                Key is the utterance-idx.

    Attributes:
        ref_outcome (Outcome): The outcome of the ground-truth/reference.
        hyp_outcome (Outcome): The outcome of the system-output/hypothesis.
        confusion (AggregatedConfusion): Confusion result
    """

    def __init__(self, ref_outcome, hyp_outcome, utt_to_segments):
        super(SegmentEvaluation, self).__init__(ref_outcome, hyp_outcome)

        self.utt_to_segments = utt_to_segments
        self.confusion = confusion.create_from_segments(self.segments)

    @property
    def default_template(self):
        return 'segment'

    @property
    def template_data(self):
        return {
            'evaluation': self,
            'ref_outcome': self.ref_outcome,
            'hyp_outcome': self.hyp_outcome,
            'utt_to_segments': self.utt_to_segments,
            'confusion': self.confusion
        }

    @property
    def segments(self):
        """ Return a list of all segment (from all utterances together). """
        all_segments = []

        for utt_segments in self.utt_to_segments.values():
            all_segments.extend(utt_segments)

        return all_segments


class SegmentEvaluator(evaluator.Evaluator):
    """
    Evaluation of an alignment based on segments.

    Arguments:
        aligner (SegmentAligner): An instance of an event-aligner to use.
                                  If not given, the :class:`alignment.InvariantSegmentAligner` is used.
    """

    def __init__(self, aligner=None):
        if aligner is None:
            self.aligner = alignment.InvariantSegmentAligner()
        else:
            self.aligner = aligner

    @classmethod
    def default_label_list_idx(cls):
        return 'domain'

    def create_alignment(self, ref, hyp):
        utt_segments = {}

        for key, ll_ref in ref.label_lists.items():
            ll_hyp = hyp.label_lists[key]

            aligned_segments = self.aligner.align(ll_ref, ll_hyp)
            aligned_segments = SegmentEvaluator.flatten_overlapping_labels(aligned_segments)

            utt_segments[key] = aligned_segments

        return utt_segments

    def do_evaluate(self, ref, hyp):
        utt_segments = self.create_alignment(ref, hyp)
        return SegmentEvaluation(ref, hyp, utt_segments)

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
