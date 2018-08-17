import audiomate
from audiomate.corpus import assets

from evalmate.alignment import one_to_one
from evalmate import confusion

from . import base


class KWSEvaluation(base.Evaluation):
    """
    Result of an evaluation of a keyword spotting task.

    Arguments:
        aligned_labels (list): List of :py:class:`evalmate.utils.structure.LabelPair`.

    Attributes
        confusion_stats (ConfusionStats): Confusion statistics
    """

    def __init__(self, aligned_labels):
        self.aligned_labels = aligned_labels
        self.confusion_stats = confusion.create_from_label_pairs(self.aligned_labels)

    @property
    def data(self):
        return self.confusion_stats


class KWSEvaluator(base.Evaluator):
    """
    Class to retrieve evaluation results for a keyword spotting task.

    Arguments:
        aligner (OneToOneAligner): An instance of an one-to-one aligner to use. If not given the
                                   :py:class:`evalmate.alignment.one_to_one.BipartiteMatchingOneToOneAligner` is used.
    """

    def __init__(self, aligner=None):
        if aligner is None:
            self.aligner = one_to_one.BipartiteMatchingOneToOneAligner()
        else:
            self.aligner = aligner

    def evaluate(self, ref, hyp):
        """
        Create the evaluation result of the given hypothesis compared to the given reference (ground truth).
        There are different possibilities of input:

        * ref = Corpus / hyp = dict: The dict contains label-lists which are compared against the corpus.
          See ``evaluate_label_lists_against_corpus``
        * ref = LabelList / hyp = LabelList: Ref label-list is compared against the other.
          See ``evaluate_label_lists``

        Arguments:
            ref (LabelList, Corpus): A label-list, a corpus.
            hyp (LabelList, dict): A label-list, a dict.

        Returns:
            KWSEvaluation: The evaluation results.
        """

        if isinstance(ref, assets.LabelList) and isinstance(hyp, assets.LabelList):
            return self.evaluate_label_lists(ref, hyp)

        if isinstance(ref, audiomate.Corpus) and isinstance(hyp, dict):
            return self.evaluate_label_lists_against_corpus(ref, hyp)

        raise ValueError('Invalid arguments!')

    def evaluate_label_lists(self, ll_ref, ll_hyp):
        """
        Create Evaluation for ref and hyp label-list.

        Arguments:
            ref (LabelList): A label-list.
            hyp (LabelList): A label-list.

        Returns:
            KWSEvaluation: The evaluation results.
        """
        aligned_labels = self.aligner.align(ll_ref, ll_hyp)
        return KWSEvaluation(aligned_labels)

    def evaluate_label_lists_against_corpus(self, corpus, label_lists, label_list_idx='word-transcript'):
        """
        Create Evaluation for the given corpus.

        Arguments:
            corpus (Corpus): A corpus containing the reference label-lists.
            label_lists (Dict): A dictionary containing label-lists with the utterance-idx as key.
                                The utterance-idx is used to find the corresponding reference label-list in the corpus.
            label_list_idx (str): The idx of the label-lists to use as reference from the corpus.
                                  By default `word-transcript` is used which mostly likely is used for kws.

        Returns:
            KWSEvaluation: The evaluation results.
        """

        aligned_labels = []

        for utterance in corpus.utterances.values():
            ll_ref = utterance.label_lists[label_list_idx]
            ll_hyp = label_lists[utterance.idx]

            aligned_labels.extend(self.aligner.align(ll_ref, ll_hyp))

        return KWSEvaluation(aligned_labels)
