from audiomate import annotations

from evalmate import alignment

from . import event


class ASREvaluation(event.EventEvaluation):
    """
    Result of an evaluation of a automatic speech recognition task.

    Arguments:
        utt_to_label_pairs (dict) Dict containing the alignment for every utterance.
                                  Key is the utterance-id, value is a list of :py:class:`evalmate.alignment.LabelPair`.

    Attributes:
        ref_outcome (Outcome): The outcome of the ground-truth/reference.
        hyp_outcome (Outcome): The outcome of the system-output/hypothesis.
        confusion (AggregatedConfusion): Confusion statistics
    """

    @property
    def default_template(self):
        return 'asr'


class ASREvaluator(event.EventEvaluator):
    """
    Class to retrieve evaluation results for a automatic speech recognition task.

    Arguments:
        aligner (EventAligner): An instance of an event-aligner to use.
                                If not given, the :class:`alignment.LevenshteinAligner` is used.
    """

    def __init__(self, aligner=None):
        if aligner is None:
            aligner = alignment.LevenshteinAligner()

        super(ASREvaluator, self).__init__(aligner)

    @classmethod
    def default_label_list_idx(cls):
        return 'word-transcript'

    def do_evaluate(self, ref, hyp):
        utt_to_label_pairs = self.create_alignment(ref, hyp)
        return ASREvaluation(ref, hyp, utt_to_label_pairs)

    def create_alignment(self, ref, hyp):
        utt_to_label_pairs = {}

        for utterance_idx, ll_ref in ref.label_lists.items():
            ll_hyp = hyp.label_lists[utterance_idx]

            ref_tokens = ASREvaluator.tokenize(ll_ref)
            hyp_tokens = ASREvaluator.tokenize(ll_hyp)

            utt_to_label_pairs[utterance_idx] = self.aligner.align(ref_tokens, hyp_tokens)

        return utt_to_label_pairs

    @staticmethod
    def tokenize(ll, overlap_threshold=0.1):
        """
        Tokenize a label-list and return a new label-list with a separate label for every token.
        """
        # TODO: CAN be replaced with ``audiomate.annotations.LabelList.tokenized()`` as soon as released.

        sorted_by_start = sorted(ll.labels)
        tokens = []
        last_label_end = None

        for label in sorted_by_start:
            if last_label_end is None or (last_label_end - label.start < overlap_threshold and last_label_end > 0):
                label_tokens = label.value.split(sep=' ')
                label_tokens = [t.strip() for t in label_tokens]

                while '' in label_tokens:
                    label_tokens.remove('')

                tokens.extend(label_tokens)
                last_label_end = label.end
            else:
                raise ValueError('Labels overlap, not able to define the correct order')

        return annotations.LabelList(labels=[annotations.Label(t) for t in tokens])
