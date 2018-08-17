def overlap_percentage(ref, hyp):
    """ Calculate the percentage of overlapping between the two labels, relative to the length of the ref-label. """

    if ref.duration <= 0:
        return 0

    start_overlap = max(ref.start, hyp.start)
    end_overlap = min(ref.end, hyp.end)

    return max(0, end_overlap - start_overlap) / ref.duration
