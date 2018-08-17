import audiomate
from audiomate.corpus import assets

from evalmate import confusion
from evalmate.alignment import segments

from . import base


class ClassificationEvaluation(base.Evaluation):
    """
    Result of an evaluation of a keyword spotting task.

    Arguments:
        aligned_segments (list): List of :py:class:`evalmate.utils.structure.Segment`.

    Attributes
        confusion (AggregatedConfusion): Confusion result
    """

    def __init__(self, aligned_segments):
        self.aligned_segments = aligned_segments
        self.confusion = confusion.create_from_segments(self.aligned_segments)

    @property
    def data(self):
        return self.confusion


class ClassificationEvaluator(base.Evaluator):
    """
    Evaluation of a sequence classification task.
    In a sequence classification task every part of a sequence has to be assigned one out of some predefined classes.

    For example imagine a recording of a radio broadcast and
    the system has to classify every part of the signal either into music or speech.
    """

    def __init__(self):
        self.aligner = segments.SegmentAligner()

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
            ClassificationEvaluation: The evaluation results.
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
            ClassificationEvaluation: The evaluation results.
        """

        aligned_segments = self.aligner.align(ll_ref, ll_hyp)
        aligned_segments = ClassificationEvaluator.flatten_overlapping_labels(aligned_segments)
        return ClassificationEvaluation(aligned_segments)

    def evaluate_label_lists_against_corpus(self, corpus, label_lists, label_list_idx='domain'):
        """
        Create Evaluation for the given corpus.

        Arguments:
            corpus (Corpus): A corpus containing the reference label-lists.
            label_lists (Dict): A dictionary containing label-lists with the utterance-idx as key.
                                The utterance-idx is used to find the corresponding reference label-list in the corpus.
            label_list_idx (str): The idx of the label-lists to use as reference from the corpus.
                                  By default `word-transcript` is used which mostly likely is used for kws.

        Returns:
            ClassificationEvaluation: The evaluation results.
        """

        all = []

        for utterance in corpus.utterances.values():
            ll_ref = utterance.label_lists[label_list_idx]
            ll_hyp = label_lists[utterance.idx]

            aligned_segments = self.aligner.align(ll_ref, ll_hyp)
            aligned_segments = ClassificationEvaluator.flatten_overlapping_labels(aligned_segments)

            all.extend(aligned_segments)

        return ClassificationEvaluation(all)

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
