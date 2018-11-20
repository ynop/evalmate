import numpy as np
import scipy

from evalmate.utils import labellist
from evalmate.utils import label

from . import utils
from . import aligner


class BipartiteMatchingAligner(aligner.EventAligner):
    """
    Create event-based alignment, based on bipartite matching.

    1. In a first step for every possible label-pair between ref and hyp,
    it is decided if a mapping of such a pair is possible.
    This decision is based on the ``start_delta_threshold`` and ``end_delta_threshold``.

    2. Using penalty and weight parameters, for every pair a penalty is computed for aligning the pair.

    3. From all the pairs and the computed probabilities, the best alignment is computed using bipartite matching.
    So that every label only occurs once in the final alignment.

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
        Return an alignment between the events of the given label-lists.

        Args:
            ref (audiomate.corpus.assets.LabelList): The label-list containing labels (events) of the ground truth.
            hyp (audiomate.corpus.assets.LabelList): The label-list containing labels (events) of the system output.

        Returns:
            list: A list of :class:`evalmate.alignment.LabelPair`. Every pair contains one label (event) from
            the ground truth and one from the system output, that are aligned. One of them also can be ``None``.
        """

        if len(ll_ref) == 0 and len(ll_hyp) == 0:
            return []

        if len(ll_ref) == 0:
            return [utils.LabelPair(None, x) for x in ll_hyp]

        if len(ll_hyp) == 0:
            return [utils.LabelPair(x, None) for x in ll_ref]

        close_pairs, ref_no_match, hyp_no_match = labellist.close_pairs(ll_ref, ll_hyp,
                                                                        self.start_delta_threshold,
                                                                        self.end_delta_threshold)

        # Calculate a high penalty for invalid matches
        invalid_penalty = self.non_overlap_penalty_weight + self.insertion_penalty + self.deletion_penalty

        # Cost matrix: Add possible insertion/deletion rows/cols
        size = len(ll_ref) + len(ll_hyp)
        cost = np.full((size, size), invalid_penalty).astype(np.float)
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

                matching.append(utils.LabelPair(ref_label, hyp_label))

        return matching


class FullMatchingAligner(aligner.EventAligner):
    """
    Event-based alignment, where all possible matches are returned.
    So a single label can occur multiple times, but with a different counterpart.

    Arguments:
        min_overlap (float): Number of seconds the segment of overlap has to be, to align two labels.
                             If ``0``, any overlap is accepted.
    """

    def __init__(self, min_overlap=0):
        self.min_overlap = min_overlap

    def align(self, ref, hyp):
        """
        Return an alignment between the events of the given label-lists.

        Args:
            ref (audiomate.corpus.assets.LabelList): The label-list containing labels (events) of the ground truth.
            hyp (audiomate.corpus.assets.LabelList): The label-list containing labels (events) of the system output.

        Returns:
            list: A list of :class:`evalmate.alignment.LabelPair`. Every pair contains one label (event) from
            the ground truth and one from the system output, that are aligned. One of them also can be ``None``.
        """

        close_pairs, ref_no_match, hyp_no_match = labellist.overlapping_pairs(ref, hyp, self.min_overlap)

        pairs = []

        for ref_index, hyp_index in close_pairs:
            pair = utils.LabelPair(ref[ref_index], hyp[hyp_index])
            pairs.append(pair)

        for ref_index in ref_no_match:
            pair = utils.LabelPair(ref[ref_index], None)
            pairs.append(pair)

        for hyp_index in hyp_no_match:
            pair = utils.LabelPair(None, hyp[hyp_index])
            pairs.append(pair)

        return pairs
