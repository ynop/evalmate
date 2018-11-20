import numpy as np

from evalmate import alignment

from . import event


class KWSEvaluation(event.EventEvaluation):
    """
    Result of an evaluation of a keyword spotting task.

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
        return 'kws'

    def keywords(self):
        """
        Return a list of all keywords occurring in the reference outcome.
        """
        return self.ref_outcome.all_values

    def false_rejection_rate(self, keyword=None):
        """
        The False Rejection Rate (FRR) is the percentage of misses of all occurrences in the ground truth.
        If no keyword is given the mean FRR is calculated over all keywords.

        Args:
            keyword (str): If not None, only the FFR for this keyword is returned.

        Returns:
            float: A rate between 0 and 1
        """

        if keyword is not None:
            conf = self.confusion.instances[keyword]

            if conf.total <= 0:
                return 0.0

            return conf.false_negatives / conf.total
        else:
            per_kw = [self.false_rejection_rate(kw) for kw in self.confusion.instances.keys()]
            return np.mean(per_kw)

    def false_alarm_rate(self, keyword=None):
        """
        The False Alarm Rate (FAR) is the percentage of detections, where no keyword is according to the ground truth.
        If no keyword is given the mean FAR is calculated over all keywords.
        This rate is relative to the duration of all utterances.

        To calculate this, we need to know the number of times a keyword could be wrongly inserted.
        We assume that every keyword takes one second to approximate this value.

        Args:
            keyword (str): If not None, only the FFR for this keyword is returned.

        Returns:
            float: A rate between 0 and 1
        """
        conf = self.confusion

        if keyword is not None:
            conf = self.confusion.instances[keyword]

            false_positive_opportunities = self.ref_outcome.total_duration - conf.total
            false_positives = conf.false_positives

            return false_positives / false_positive_opportunities
        else:
            per_kw = [self.false_alarm_rate(kw) for kw in self.confusion.instances.keys()]
            return np.mean(per_kw)

    def term_weighted_value(self, keyword=None):
        """
        Computes the Term-Weighted Value (TWV).

        Note:
            The TWV is implemented according to
            `OpenKWS 2016 Evaluation Plan
            <https://www.nist.gov/sites/default/files/documents/itl/iad/mig/KWS16-evalplan-v04.pdf>`_

        Args:
            keyword (str): If None, computes the TWV over all keywords, otherwise only for the given keyword.

        Returns:
            float: The TWV in the range 1 to -inf
        """

        p_miss = self.false_rejection_rate(keyword=keyword)
        p_false_alarm = self.false_alarm_rate(keyword=keyword)

        false_alarm_cost = 0.1
        correct_cost = 1.0
        kw_prior = 0.0001

        beta = false_alarm_cost / correct_cost * (kw_prior ** -1 - 1)

        return 1 - (p_miss + beta * p_false_alarm)


class KWSEvaluator(event.EventEvaluator):
    """
    Class to retrieve evaluation results for a keyword spotting task.

    Arguments:
        aligner (EventAligner): An instance of an event-aligner to use.
                                If not given the :class:`evalmate.alignment.BipartiteMatchingAligner` is user.
    """

    def __init__(self, aligner=None):
        if aligner is None:
            aligner = alignment.BipartiteMatchingAligner()

        super(KWSEvaluator, self).__init__(aligner)

    @classmethod
    def default_label_list_idx(cls):
        return 'word-transcript'

    def do_evaluate(self, ref, hyp):
        utt_to_label_pairs = self.create_alignment(ref, hyp)
        return KWSEvaluation(ref, hyp, utt_to_label_pairs)
