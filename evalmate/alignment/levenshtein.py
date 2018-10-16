import numpy as np


class LevenshteinAligner(object):
    """
    Class for calculating edit distance between two sequences and align them.

    Args:
        deletion_cost (float): Cost for a deletion in the alignment.
        insertion_cost (float): Cost for a insertion in the alignment.
        substitution_cost (float): Cost for a substitution in the alignment.
        custom_substitution_cost_function (func): Function to calculate substitution cost depending on the elements.
                                                  The function has to take two paramters (reference, hypothesis).
    """

    def __init__(self, deletion_cost=3, insertion_cost=3, substitution_cost=4, custom_substitution_cost_function=None):
        self.deletion_cost = deletion_cost
        self.insertion_cost = insertion_cost
        self.substitution_cost = substitution_cost
        self.custom_substitution_cost_function = custom_substitution_cost_function

    def align(self, reference, hypothesis):
        """
        Return a tuple with aligned sequences. None for insertions and deletions.

        Args:
            reference (list): List of strings, representing the elements in the reference (ground truth).
            hypothesis (list): List of strings, representing the elements in the hypothesis (system output).

        Returns:
            tuple: A tuple with (ref, hyp), with two lists for ref/hyp with aligned elements.
                   The element with index x from ref corresponds to the element with index x from hyp.

        Example:
            >>> reference = ['a', 'b', 'c']
            >>> hypothesis = ['a', 'c']
            >>>
            >>> Levenshtein()
            ( ['a', 'b', 'c'], ['a', None, 'c'] )
        """
        dist_mat = self._calc_distance_matrix(reference, hypothesis)

        n_ref = len(reference) + 1
        n_hyp = len(hypothesis) + 1

        ref_aligned = []
        hyp_aligned = []

        i = n_ref - 1
        j = n_hyp - 1

        ref_index = len(reference) - 1
        hyp_index = len(hypothesis) - 1

        while i > 0 or j > 0:
            if j > 0 and dist_mat[i, j - 1] + self.insertion_cost == dist_mat[i, j]:
                ref_aligned.insert(0, None)
                hyp_aligned.insert(0, hypothesis[hyp_index])
                hyp_index -= 1
                j -= 1
            elif i > 0 and dist_mat[i - 1, j] + self.deletion_cost == dist_mat[i, j]:
                ref_aligned.insert(0, reference[ref_index])
                ref_index -= 1
                hyp_aligned.insert(0, None)
                i -= 1
            else:
                ref_aligned.insert(0, reference[ref_index])
                ref_index -= 1
                hyp_aligned.insert(0, hypothesis[hyp_index])
                hyp_index -= 1
                i -= 1
                j -= 1

        return ref_aligned, hyp_aligned

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
        if ref_element == hyp_element:
            return 0
        elif self.custom_substitution_cost_function is None:
            return self.substitution_cost
        else:
            return self.custom_substitution_cost_function(ref_element, hyp_element)
