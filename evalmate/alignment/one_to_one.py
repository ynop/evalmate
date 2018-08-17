import abc

import numpy as np
import scipy

from evalmate.utils import labellist
from evalmate.utils import label
from evalmate.utils import structure


class OneToOneAligner(abc.ABC):
    """
    Base class for aligning labels one-to-one.
    The result is a list of pairs (ref to hyp label).
    Not all labels have to be in a pair.
    """

    @abc.abstractmethod
    def align(self, ll_ref, ll_hyp):
        """
        Align two ref and hyp label-list.

        Arguments:
            ll_ref (LabelList): Reference labels (Ground Truth).
            ll_hyp (LabelList): Hypothesis labels.

        Returns:
            list: List of tuples (ref-index, hyp-index)
        """
        pass


class BipartiteMatchingOneToOneAligner(OneToOneAligner):
    """
    Class to create one-to-one alignments.
    The best alignment is searched based on penalties given.
    The thresholds define which label-pairs are considered for alignment in first place.

    Arguments:
        start_delta_threshold (float): Temporal tolerance of the start time in seconds.
                                       If the delta between the starts of the two labels is
                                       greater it is not a matching pair.
        end_delta_threshold (float): Temporal tolerance of the end time in seconds.
                                     If the delta between the ends of the two labels is greater
                                     it is not a matching pair. If < 0 the end time is not checked at all.
        non_overlap_penalty_weight (float): Weight-factor of penalty for the non-overlapping ratio between two labels.
        substitution_penalty (float): Penalty for aligning two labels with different values.
        deletion_penalty (float): Penalty for aligning a reference-label with no hypothesis-label.
        insertion_penalty (float): Penalty for aligning a hypothesis-label with no reference-label.
    """

    def __init__(self, start_delta_threshold=0.5, end_delta_threshold=-1, non_overlap_penalty_weight=1,
                 substitution_penalty=2, insertion_penalty=10, deletion_penalty=10):
        self.start_delta_threshold = start_delta_threshold
        self.end_delta_threshold = end_delta_threshold

        self.non_overlap_penalty_weight = non_overlap_penalty_weight
        self.substitution_penalty = substitution_penalty
        self.insertion_penalty = insertion_penalty
        self.deletion_penalty = deletion_penalty

    def align(self, ll_ref, ll_hyp):
        """
        Align two ref and hyp label-list.

        Arguments:
            ll_ref (LabelList): Reference labels (Ground Truth).
            ll_hyp (LabelList): Hypothesis labels.

        Returns:
            list: List of tuples (ref-index, hyp-index)
        """
        close_pairs, ref_no_match, hyp_no_match = labellist.close_pairs(ll_ref, ll_hyp,
                                                                        self.start_delta_threshold,
                                                                        self.end_delta_threshold)

        # Calculate a high penalty for invalid matches
        invalid_penalty = self.non_overlap_penalty_weight + self.insertion_penalty + self.deletion_penalty

        # Cost matrix: Add possible insertion/deletion rows/cols
        cost = np.full((2 * len(ll_ref), 2 * len(ll_hyp)), invalid_penalty).astype(np.float)
        cost[len(ll_ref):, :] = self.insertion_penalty
        cost[:, len(ll_hyp):] = self.deletion_penalty

        for pair in close_pairs:
            ref = ll_ref[pair[0]]
            hyp = ll_hyp[pair[1]]
            penalty = 0

            if ref.value != hyp.value:
                penalty += self.substitution_penalty

            penalty += self.non_overlap_penalty_weight * (1 - label.overlap_percentage(ref, hyp))

            cost[pair[0], pair[1]] = penalty

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

        matching = []

        for i in range(len(row_ind)):
            ref_ind = row_ind[i]
            hyp_ind = col_ind[i]

            if ref_ind >= len(ll_ref):
                ref_ind = -1

            if hyp_ind >= len(ll_hyp):
                hyp_ind = -1

            if hyp_ind != -1 or ref_ind != -1:
                if ref_ind >= 0:
                    ref_label = ll_ref[ref_ind]
                else:
                    ref_label = None

                if hyp_ind >= 0:
                    hyp_label = ll_hyp[hyp_ind]
                else:
                    hyp_label = None

                matching.append(structure.LabelPair(ref_label, hyp_label))

        return matching
