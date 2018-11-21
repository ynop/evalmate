import numpy as np

from . import utils
from . import aligner


class LevenshteinAligner(aligner.EventAligner):
    """
    Alignment of labels of two label-lists based on the Levenshtein distance
    (https://en.wikipedia.org/wiki/Levenshtein_distance).

    This only takes the order of the labels into account, not the start and end-times.

    Args:
        deletion_cost (float): Cost for a deletion in the alignment.
        insertion_cost (float): Cost for a insertion in the alignment.
        substitution_cost (float): Cost for a substitution in the alignment.
        custom_substitution_cost_function (func): Function to calculate substitution cost depending on the elements.
                                                  The function has to take two paramters (ref-label, hyp-label).
    """

    def __init__(self, deletion_cost=3, insertion_cost=3, substitution_cost=4, custom_substitution_cost_function=None):
        self.deletion_cost = deletion_cost
        self.insertion_cost = insertion_cost
        self.substitution_cost = substitution_cost
        self.custom_substitution_cost_function = custom_substitution_cost_function

    def align(self, reference, hypothesis):
        """
        Return an alignment between the labels of the given label-lists.

        Args:
            reference (audiomate.corpus.assets.LabelList): The label-list containing labels of the ground truth.
            hypothesis (audiomate.corpus.assets.LabelList): The label-list containing labels of the system output.

        Returns:
            list: A list of :class:`evalmate.alignment.LabelPair`. Every pair contains one label from
            the ground truth and one from the system output, that are aligned. One of them also can be ``None``.

        Example:

            >>> from audiomate.corpus import assets
            >>>
            >>> reference = assets.LabelList(labels=[
            >>>     assets.Label('a'),
            >>>     assets.Label('b'),
            >>>     assets.Label('c')
            >>> ])
            >>> hypothesis = assets.LabelList(labels=([
            >>>     assets.Label('a'),
            >>>     assets.Label('c')
            >>> ])
            >>>
            >>> LevenshteinAligner().align(reference, hypothesis)
            [
                LabelPair(Label('a'), Label('a')),
                LabelPair(Label('b'), None),
                LabelPair(Label('c'), Label('c'))
            ]
        """
        dist_mat = self._calc_distance_matrix(reference, hypothesis)

        n_ref = len(reference) + 1
        n_hyp = len(hypothesis) + 1

        aligned_pairs = []

        i = n_ref - 1
        j = n_hyp - 1

        ref_index = len(reference) - 1
        hyp_index = len(hypothesis) - 1

        while i > 0 or j > 0:
            if j > 0 and dist_mat[i, j - 1] + self.insertion_cost == dist_mat[i, j]:
                pair = utils.LabelPair(None, hypothesis[hyp_index])
                aligned_pairs.insert(0, pair)
                hyp_index -= 1
                j -= 1
            elif i > 0 and dist_mat[i - 1, j] + self.deletion_cost == dist_mat[i, j]:
                pair = utils.LabelPair(reference[ref_index], None)
                aligned_pairs.insert(0, pair)
                ref_index -= 1
                i -= 1
            else:
                pair = utils.LabelPair(reference[ref_index], hypothesis[hyp_index])
                aligned_pairs.insert(0, pair)
                ref_index -= 1
                hyp_index -= 1
                i -= 1
                j -= 1

        return aligned_pairs

    def calculate_edit_distance(self, reference, hypothesis):
        dist_mat = self._calc_distance_matrix(reference, hypothesis)

        return dist_mat[len(reference), len(hypothesis)]

    def _calc_distance_matrix(self, reference, hypothesis):
        """ Calculate the distance matrix between two sequences. """
        n_ref = len(reference) + 1
        n_hyp = len(hypothesis) + 1

        mat = np.zeros((n_ref, n_hyp), dtype=np.int16)

        for i in range(1, n_ref):
            mat[i, 0] = i * self.deletion_cost

        for j in range(1, n_hyp):
            mat[0, j] = j * self.insertion_cost

        for i in range(1, n_ref):
            for j in range(1, n_hyp):
                sub_cost = self._get_substitution_cost(reference[i - 1], hypothesis[j - 1])

                ops = [
                    mat[i - 1, j - 1] + sub_cost,  # correct / substitution
                    mat[i, j - 1] + self.insertion_cost,  # insertion
                    mat[i - 1, j] + self.deletion_cost  # deletion
                ]

                mat[i, j] = min(ops)

        return mat

    def _get_substitution_cost(self, ref_element, hyp_element):
        if ref_element.value == hyp_element.value:
            return 0
        elif self.custom_substitution_cost_function is None:
            return self.substitution_cost
        else:
            return self.custom_substitution_cost_function(ref_element, hyp_element)
