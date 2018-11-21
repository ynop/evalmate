from evalmate.utils import label


def close_pairs(ll_ref, ll_hyp, start_delta_threshold=0.5, end_delta_threshold=-1):
    """
    Return a list of all close label pairs as indices within label-list.
    Close means within a given time threshold.

    Args:
        ll_ref (LabelList): LabelList with reference/ground-truth labels.
        ll_hyp (LabelList): LabelList with hypothesis labels.
        start_delta_threshold (float): Temporal tolerance of the start time in seconds.
                                       If the delta between the starts of the two labels is
                                       greater it is not a matching pair.
        end_delta_threshold (float): Temporal tolerance of the end time in seconds.
                                     If the delta between the ends of the two labels is greater
                                     it is not a matching pair. If < 0 the end time is not checked at all.

    Returns:
        list, set, set: List of tuples (ref-index, hyp-index), Set containing indices of ref-labels with no match,
                        Set containing indices of hyp-labels with no match.
    """

    matches = []
    ref_no_match = set(range(len(ll_ref)))
    hyp_no_match = set(range(len(ll_hyp)))

    for ref_index, ref in enumerate(ll_ref):

        for hyp_index, hyp in enumerate(ll_hyp):
            start_delta = abs(ref.start - hyp.start)

            if start_delta <= start_delta_threshold:

                if end_delta_threshold < 0.0 or abs(ref.end - hyp.end) < end_delta_threshold:
                    matches.append((ref_index, hyp_index))

                    if hyp_index in hyp_no_match:
                        hyp_no_match.remove(hyp_index)

                    if ref_index in ref_no_match:
                        ref_no_match.remove(ref_index)

    return matches, ref_no_match, hyp_no_match


def overlapping_pairs(ll_ref, ll_hyp, min_overlap=0):
    """
    Return a list of all label pairs that overlap. If a label overlaps with multiple labels,
    a pair for every combination is returned.

    Args:
        ll_ref (LabelList): LabelList with reference/ground-truth labels.
        ll_hyp (LabelList): LabelList with hypothesis labels.
        min_overlap (float): Number of seconds the segment of overlap has to be, to include the combination of labels.
                             If ``0``, any overlap is accepted.

    Returns:
        list, set, set: List of tuples (ref-index, hyp-index), Set containing indices of ref-labels with no match,
                        Set containing indices of hyp-labels with no match.
    """

    matches = []
    ref_no_match = set(range(len(ll_ref)))
    hyp_no_match = set(range(len(ll_hyp)))

    for ref_index, ref in enumerate(ll_ref):
        for hyp_index, hyp in enumerate(ll_hyp):
            ref = ll_ref[ref_index]
            hyp = ll_hyp[hyp_index]

            overlap_time = label.overlap_time(ref, hyp)

            if min_overlap <= 0 or overlap_time >= min_overlap:
                matches.append((ref_index, hyp_index))

                if hyp_index in hyp_no_match:
                    hyp_no_match.remove(hyp_index)

                if ref_index in ref_no_match:
                    ref_no_match.remove(ref_index)

    return matches, ref_no_match, hyp_no_match
