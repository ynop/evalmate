import abc

from evalmate.utils import label


class CandidateFinder(abc.ABC):
    """
    Class to find possible pairs of labels for further alignment.
    This is used for preprocessing and finding pairs of labels
    that may be aligned together. A label can be a candidate in multiple pairs.
    """

    @abc.abstractmethod
    def find(self, ref_ll, hyp_ll):
        """
        Return candidates as pairs of labels, as well as labels that
        have no possible counterparts.

        Args:
            ref_ll (LabelList): LabelList with reference (ground truth).
            hyp_ll (LabelList): LabelList with hypothesis (system output).

        Returns:
            tuple: A tuple (candidates, single-ref, single-hyp) containing
            the candidates in paris, the ref-labels and the hyp-labels,
            that have no possible counterpart.
        """
        raise NotImplementedError()


class StartEndCandidateFinder(CandidateFinder):
    """
    Finds candidates based on the difference between the start (and end)
    of two labels for a possible pairs.

    Args:
        start_delta_threshold (float): Temporal tolerance of the start
        time in seconds. If the delta between the starts of the two labels is
        greater it is not a matching pair.
        end_delta_threshold (float): Temporal tolerance of the end time in
        seconds. If the delta between the ends of the two labels is greater
        it is not a matching pair. If < 0 the end time is not checked at all.

    """

    def __init__(self, start_delta_threshold, end_delta_threshold=-1):
        self.start_delta_threshold = start_delta_threshold
        self.end_delta_threshold = end_delta_threshold

    def find(self, ref_ll, hyp_ll):
        matches = []

        ref_no_match = set(range(len(ref_ll)))
        hyp_no_match = set(range(len(hyp_ll)))

        for ref_index, ref in enumerate(ref_ll):

            for hyp_index, hyp in enumerate(hyp_ll):
                start_delta = abs(ref.start - hyp.start)

                if start_delta <= self.start_delta_threshold:

                    if self.end_delta_threshold < 0.0 or \
                            abs(ref.end - hyp.end) < self.end_delta_threshold:
                        matches.append((ref_index, hyp_index))

                        if hyp_index in hyp_no_match:
                            hyp_no_match.remove(hyp_index)

                        if ref_index in ref_no_match:
                            ref_no_match.remove(ref_index)

        return matches, ref_no_match, hyp_no_match


class OverlapCandidateFinder(CandidateFinder):
    """
    Finds candidates based on amount of overlapping between two labels.

    Args:
        min_overlap (float): Number of seconds the segment of overlap has to
        be, to include the combination of labels. (default 0.05 seconds)
    """

    def __init__(self, min_overlap=0.05):
        self.min_overlap = 0.05

    def find(self, ref_ll, hyp_ll):
        matches = []

        ref_no_match = set(range(len(ref_ll)))
        hyp_no_match = set(range(len(hyp_ll)))

        for ref_index, ref in enumerate(ref_ll):
            for hyp_index, hyp in enumerate(hyp_ll):
                ref = ref_ll[ref_index]
                hyp = hyp_ll[hyp_index]

                overlap_time = label.overlap_time(ref, hyp)

                if self.min_overlap <= 0 or overlap_time >= self.min_overlap:
                    matches.append((ref_index, hyp_index))

                    if hyp_index in hyp_no_match:
                        hyp_no_match.remove(hyp_index)

                    if ref_index in ref_no_match:
                        ref_no_match.remove(ref_index)

        return matches, ref_no_match, hyp_no_match
