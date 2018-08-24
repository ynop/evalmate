from evalmate.alignment import one_to_one
from evalmate import confusion

from . import base


class KWSEvaluation(base.Evaluation):
    """
    Result of an evaluation of a keyword spotting task.

    Arguments:
        aligned_labels (list): List of :py:class:`evalmate.utils.structure.LabelPair`.

    Attributes
        ref_outcome (Outcome): The outcome of the ground-truth/reference.
        hyp_outcome (Outcome): The outcome of the system-output/hypothesis.
        confusion_stats (ConfusionStats): Confusion statistics
    """

    def __init__(self, ref_outcome, hyp_outcome, aligned_labels):
        super(KWSEvaluation, self).__init__(ref_outcome, hyp_outcome)
        self.aligned_labels = aligned_labels
        self.confusion_stats = confusion.create_from_label_pairs(self.aligned_labels)

    @property
    def default_template(self):
        return 'kws'

    @property
    def template_data(self):
        return {'confusion': self.confusion_stats}


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

    @classmethod
    def default_label_list_idx(cls):
        return 'word-transcript'

    def do_evaluate(self, ref, hyp):
        aligned_labels = []

        for key, ll_ref in ref.label_lists.items():
            ll_hyp = hyp.label_lists[key]
            aligned_labels.extend(self.aligner.align(ll_ref, ll_hyp))

        return KWSEvaluation(ref, hyp, aligned_labels)
