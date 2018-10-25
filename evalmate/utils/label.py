def overlap_percentage(ref, hyp):
    """ Calculate the percentage of overlapping between the two labels, relative to the length of the ref-label. """

    if ref.duration <= 0:
        return 0

    return overlap_time(ref, hyp) / ref.duration


def overlap_time(ref, hyp):
    """ Calculate the number of seconds of the overlapping between the two lables. """

    ref_end = ref.end
    hyp_end = hyp.end

    if ref_end == -1:
        ref_end = hyp_end

    if hyp_end == -1:
        hyp_end = ref_end

    start_overlap = max(ref.start, hyp.start)
    end_overlap = min(ref_end, hyp_end)

    return max(0, end_overlap - start_overlap)
