from audiomate.corpus import assets

from evalmate import confusion
from evalmate import alignment
from evalmate.utils import structure

from . import base


class ASREvaluation(base.Evaluation):
    """
    Result of an evaluation of a automatic speech recognition task.

    Arguments:
        aligned_tokens (dict): Dictionary container for every utterance a tuple.
                               The tuple contains the aligned tokens (e.g words, phones).
                               The first list in the tuple is the reference, the second the hypothesis.

    Attributes:
        ref_outcome (Outcome): The outcome of the ground-truth/reference.
        hyp_outcome (Outcome): The outcome of the system-output/hypothesis.
        confusion (AggregatedConfusion): Confusion statistics
    """

    def __init__(self, ref_outcome, hyp_outcome, aligned_tokens):
        super(ASREvaluation, self).__init__(ref_outcome, hyp_outcome)
        self.aligned_tokens = aligned_tokens

        label_pairs = []

        for utt_idx, aligned in aligned_tokens.items():

            if len(aligned[0]) != len(aligned[1]):
                raise ValueError('Ref and hyp alignment of utt {} have not the same length!'.format(utt_idx))

            for ref, hyp in zip(*aligned):
                ref_label = None
                hyp_label = None

                if ref is not None:
                    ref_label = assets.Label(ref)

                if hyp is not None:
                    hyp_label = assets.Label(hyp)

                pair = structure.LabelPair(ref_label, hyp_label)
                label_pairs.append(pair)

        self.confusion = confusion.create_from_label_pairs(label_pairs)

    @property
    def default_template(self):
        return 'asr'

    @property
    def template_data(self):
        return {
            'confusion': self.confusion,
            'aligned_tokens': self.aligned_tokens,
            'eval': self
        }


class ASREvaluator(base.Evaluator):
    """
    Class to retrieve evaluation results for a automatic speech recognition task.
    """

    @classmethod
    def default_label_list_idx(cls):
        return 'word-transcript'

    def do_evaluate(self, ref, hyp):
        aligned_tokens = {}
        aligner = alignment.LevenshteinAligner()

        for key, ll_ref in ref.label_lists.items():
            ll_hyp = hyp.label_lists[key]
            ref_tokens = ASREvaluator.tokenize(ll_ref)
            hyp_tokens = ASREvaluator.tokenize(ll_hyp)

            aligned_tokens[key] = aligner.align(ref_tokens, hyp_tokens)

        return ASREvaluation(ref, hyp, aligned_tokens)

    @staticmethod
    def tokenize(ll, overlap_threshold=0.1):
        """
        Tokenize a label-list.
        """
        # TODO: CAN be replaced with ``audiomate.corpus.assets.LabelList.tokenized()`` as soon as released.

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

        return tokens
