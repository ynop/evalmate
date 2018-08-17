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
