from evalmate import confusion

from . import evaluator


class EventEvaluation(evaluator.Evaluation):
    """
    Result of an evaluation of any event-based alignment.

    Arguments:
        utt_to_label_pairs (dict) Dict containing the alignment for every utterance.
                                  Key is the utterance-id, value is a list of :py:class:`evalmate.alignment.LabelPair`.

    Attributes:
        ref_outcome (Outcome): The outcome of the ground-truth/reference.
        hyp_outcome (Outcome): The outcome of the system-output/hypothesis.
        confusion (AggregatedConfusion): Confusion statistics
    """

    def __init__(self, ref_outcome, hyp_outcome, utt_to_label_pairs):
        super(EventEvaluation, self).__init__(ref_outcome, hyp_outcome)
        self.utt_to_label_pairs = utt_to_label_pairs
        self.confusion = confusion.create_from_label_pairs(self.label_pairs)

    @property
    def default_template(self):
        return 'event'

    @property
    def template_data(self):
        return {
            'evaluation': self,
            'ref_outcome': self.ref_outcome,
            'hyp_outcome': self.hyp_outcome,
            'utt_to_label_pairs': self.utt_to_label_pairs,
            'label_pairs': self.label_pairs,
            'confusion': self.confusion
        }

    @property
    def label_pairs(self):
        """
        Return a list of all label-pairs (from all utterances together).
        """
        lp = []

        for pairs in self.utt_to_label_pairs.values():
            lp.extend(pairs)

        return lp


class EventEvaluator(evaluator.Evaluator):
    """
    Class to compute evaluation results for any event-based alignment.

    Arguments:
        aligner (EventAligner): An instance of an event-aligner to use.
    """

    def __init__(self, aligner):
        self.aligner = aligner

    @classmethod
    def default_label_list_idx(cls):
        return 'word-transcript'

    def do_evaluate(self, ref, hyp):
        utt_to_label_pairs = self.create_alignment(ref, hyp)
        return EventEvaluation(ref, hyp, utt_to_label_pairs)

    def create_alignment(self, ref, hyp):
        utt_to_label_pairs = {}

        for utterance_idx, ll_ref in ref.label_lists.items():
            ll_hyp = hyp.label_lists[utterance_idx]

            utt_to_label_pairs[utterance_idx] = self.aligner.align(ll_ref, ll_hyp)

        return utt_to_label_pairs
