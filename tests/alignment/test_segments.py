from audiomate.corpus import assets

from evalmate.alignment import segments

import pytest


@pytest.fixture
def aligner():
    return segments.SegmentAligner()


class TestSegmentAligner:

    def test_align(self, aligner):
        ref = assets.LabelList(labels=[
            assets.Label('a', 0, 3),
            assets.Label('b', 3, 6),
            assets.Label('c', 7, 10)
        ])

        hyp = assets.LabelList(labels=[
            assets.Label('a', 0, 3),
            assets.Label('b', 4, 8),
            assets.Label('c', 8, 10)
        ])

        alignment = aligner.align(ref, hyp)

        assert len(alignment) == 6

        segment = alignment[0]
        assert segment.start == 0
        assert segment.end == 3
        assert segment.ref == [assets.Label('a', 0, 3)]
        assert segment.hyp == [assets.Label('a', 0, 3)]

        segment = alignment[1]
        assert segment.start == 3
        assert segment.end == 4
        assert segment.ref == [assets.Label('b', 3, 6)]
        assert segment.hyp == []

        segment = alignment[2]
        assert segment.start == 4
        assert segment.end == 6
        assert segment.ref == [assets.Label('b', 3, 6)]
        assert segment.hyp == [assets.Label('b', 4, 8)]

        segment = alignment[3]
        assert segment.start == 6
        assert segment.end == 7
        assert segment.ref == []
        assert segment.hyp == [assets.Label('b', 4, 8)]

        segment = alignment[4]
        assert segment.start == 7
        assert segment.end == 8
        assert segment.ref == [assets.Label('c', 7, 10)]
        assert segment.hyp == [assets.Label('b', 4, 8)]

        segment = alignment[5]
        assert segment.start == 8
        assert segment.end == 10
        assert segment.ref == [assets.Label('c', 7, 10)]
        assert segment.hyp == [assets.Label('c', 8, 10)]

    def test_align_with_overlapping_labels(self, aligner):
        ref = assets.LabelList(labels=[
            assets.Label('a', 0, 3),
            assets.Label('b', 3, 6),
            assets.Label('bx', 4, 5),
            assets.Label('c', 7, 10)
        ])

        hyp = assets.LabelList(labels=[
            assets.Label('a', 0, 3),
            assets.Label('b', 4, 8),
            assets.Label('c', 8, 10),
            assets.Label('cx', 9, 11)
        ])

        alignment = aligner.align(ref, hyp)

        assert len(alignment) == 9

        segment = alignment[0]
        assert segment.start == 0
        assert segment.end == 3
        assert segment.ref == [assets.Label('a', 0, 3)]
        assert segment.hyp == [assets.Label('a', 0, 3)]

        segment = alignment[1]
        assert segment.start == 3
        assert segment.end == 4
        assert segment.ref == [assets.Label('b', 3, 6)]
        assert segment.hyp == []

        segment = alignment[2]
        assert segment.start == 4
        assert segment.end == 5
        assert segment.ref == [assets.Label('b', 3, 6), assets.Label('bx', 4, 5)]
        assert segment.hyp == [assets.Label('b', 4, 8)]

        segment = alignment[3]
        assert segment.start == 5
        assert segment.end == 6
        assert segment.ref == [assets.Label('b', 3, 6)]
        assert segment.hyp == [assets.Label('b', 4, 8)]

        segment = alignment[4]
        assert segment.start == 6
        assert segment.end == 7
        assert segment.ref == []
        assert segment.hyp == [assets.Label('b', 4, 8)]

        segment = alignment[5]
        assert segment.start == 7
        assert segment.end == 8
        assert segment.ref == [assets.Label('c', 7, 10)]
        assert segment.hyp == [assets.Label('b', 4, 8)]

        segment = alignment[6]
        assert segment.start == 8
        assert segment.end == 9
        assert segment.ref == [assets.Label('c', 7, 10)]
        assert segment.hyp == [assets.Label('c', 8, 10)]

        segment = alignment[7]
        assert segment.start == 9
        assert segment.end == 10
        assert segment.ref == [assets.Label('c', 7, 10)]
        assert segment.hyp == [assets.Label('c', 8, 10), assets.Label('cx', 9, 11)]

        segment = alignment[8]
        assert segment.start == 10
        assert segment.end == 11
        assert segment.ref == []
        assert segment.hyp == [assets.Label('cx', 9, 11)]

    def test_align_with_empty_segments(self, aligner):
        ref = assets.LabelList(labels=[
            assets.Label('a', 0, 3),
            assets.Label('b', 4, 6),
        ])

        hyp = assets.LabelList(labels=[
            assets.Label('a', 0, 3),
            assets.Label('c', 5, 8),
        ])

        alignment = aligner.align(ref, hyp)

        assert len(alignment) == 5

        segment = alignment[0]
        assert segment.start == 0
        assert segment.end == 3
        assert segment.ref == [assets.Label('a', 0, 3)]
        assert segment.hyp == [assets.Label('a', 0, 3)]

        segment = alignment[1]
        assert segment.start == 3
        assert segment.end == 4
        assert segment.ref == []
        assert segment.hyp == []

        segment = alignment[2]
        assert segment.start == 4
        assert segment.end == 5
        assert segment.ref == [assets.Label('b', 4, 6)]
        assert segment.hyp == []

        segment = alignment[3]
        assert segment.start == 5
        assert segment.end == 6
        assert segment.ref == [assets.Label('b', 4, 6)]
        assert segment.hyp == [assets.Label('c', 5, 8)]

        segment = alignment[4]
        assert segment.start == 6
        assert segment.end == 8
        assert segment.ref == []
        assert segment.hyp == [assets.Label('c', 5, 8)]

    def test_align_empty_ground_truth(self, aligner):
        ref = assets.LabelList(labels=[
        ])

        hyp = assets.LabelList(labels=[
            assets.Label('b', 4, 8)
        ])

        alignment = aligner.align(ref, hyp)

        assert len(alignment) == 1

        segment = alignment[0]
        assert segment.start == 4
        assert segment.end == 8
        assert segment.ref == []
        assert segment.hyp == [assets.Label('b', 4, 8)]

    def test_align_empty_hypothesis(self, aligner):
        ref = assets.LabelList(labels=[
            assets.Label('b', 4, 8)
        ])

        hyp = assets.LabelList(labels=[
        ])

        alignment = aligner.align(ref, hyp)

        assert len(alignment) == 1

        segment = alignment[0]
        assert segment.start == 4
        assert segment.end == 8
        assert segment.ref == [assets.Label('b', 4, 8)]
        assert segment.hyp == []

    def test_align_different_start(self, aligner):
        ref = assets.LabelList(labels=[
            assets.Label('b', 4, 9)
        ])

        hyp = assets.LabelList(labels=[
            assets.Label('b', 2, 8)
        ])

        alignment = aligner.align(ref, hyp)

        assert len(alignment) == 3

        segment = alignment[0]
        assert segment.start == 2
        assert segment.end == 4
        assert segment.ref == []
        assert segment.hyp == [assets.Label('b', 2, 8)]

        segment = alignment[1]
        assert segment.start == 4
        assert segment.end == 8
        assert segment.ref == [assets.Label('b', 4, 9)]
        assert segment.hyp == [assets.Label('b', 2, 8)]

        segment = alignment[2]
        assert segment.start == 8
        assert segment.end == 9
        assert segment.ref == [assets.Label('b', 4, 9)]
        assert segment.hyp == []

    def test_create_event_list(self):
        ll_ref = assets.LabelList(labels=[
            assets.Label('a', 0.89, 13.73),
            assets.Label('a', 13.73, 17.49),
            assets.Label('b', 17.49, 22.75)
        ])

        ll_hyp = assets.LabelList(labels=[
            assets.Label('b', 0.1, 1.656),
            assets.Label('a', 1.656, 1.976),
            assets.Label('b', 1.976, 3.896),
            assets.Label('a', 3.896, 3.957)
        ])

        events = segments.SegmentAligner.create_event_list(ll_ref, ll_hyp, time_threshold=0.01)

        assert events[0] == (0.1, [(0.1, 'S', 1, assets.Label('b', 0.1, 1.656))])
        assert events[1] == (0.89, [(0.89, 'S', 0, assets.Label('a', 0.89, 13.73))])
        assert events[2] == (1.656, [(1.656, 'E', 1, assets.Label('b', 0.1, 1.656)),
                                     (1.656, 'S', 1, assets.Label('a', 1.656, 1.976))])
        assert events[3] == (1.976, [(1.976, 'E', 1, assets.Label('a', 1.656, 1.976)),
                                     (1.976, 'S', 1, assets.Label('b', 1.976, 3.896))])
        assert events[4] == (3.896, [(3.896, 'E', 1, assets.Label('b', 1.976, 3.896)),
                                     (3.896, 'S', 1, assets.Label('a', 3.896, 3.957))])
        assert events[5] == (3.957, [(3.957, 'E', 1, assets.Label('a', 3.896, 3.957))])
        assert events[6] == (13.73, [(13.73, 'E', 0, assets.Label('a', 0.89, 13.73)),
                                     (13.73, 'S', 0, assets.Label('a', 13.73, 17.49))])
        assert events[7] == (17.49, [(17.49, 'E', 0, assets.Label('a', 13.73, 17.49)),
                                     (17.49, 'S', 0, assets.Label('b', 17.49, 22.75))])
        assert events[8] == (22.75, [(22.75, 'E', 0, assets.Label('b', 17.49, 22.75))])
