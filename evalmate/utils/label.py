def overlap_percentage(ref, hyp):
    """ Calculate the percentage of overlapping between the two labels, relative to the length of the ref-label. """

    if ref.duration <= 0:
        return 0

    return overlap_time(ref, hyp) / ref.duration


def overlap_time(ref, hyp):
    """ Calculate the number of seconds of the overlapping between the two lables. """

    start_overlap = max(ref.start, hyp.start)
    end_overlap = min(ref.end, hyp.end)

    return max(0, end_overlap - start_overlap)
