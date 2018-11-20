from audiomate import annotations

from evalmate import alignment


class TestInvariantSegmentAligner:

    def test_align(self):
        ref = annotations.LabelList(labels=[
            annotations.Label('a', 0, 3),
            annotations.Label('b', 3, 6),
            annotations.Label('c', 7, 10)
        ])

        hyp = annotations.LabelList(labels=[
            annotations.Label('a', 0, 3),
            annotations.Label('b', 4, 8),
            annotations.Label('c', 8, 10)
        ])

        result = alignment.InvariantSegmentAligner().align(ref, hyp)

        assert len(result) == 6

        segment = result[0]
        assert segment.start == 0
        assert segment.end == 3
        assert segment.ref == [annotations.Label('a', 0, 3)]
        assert segment.hyp == [annotations.Label('a', 0, 3)]

        segment = result[1]
        assert segment.start == 3
        assert segment.end == 4
        assert segment.ref == [annotations.Label('b', 3, 6)]
        assert segment.hyp == []

        segment = result[2]
        assert segment.start == 4
        assert segment.end == 6
        assert segment.ref == [annotations.Label('b', 3, 6)]
        assert segment.hyp == [annotations.Label('b', 4, 8)]

        segment = result[3]
        assert segment.start == 6
        assert segment.end == 7
        assert segment.ref == []
        assert segment.hyp == [annotations.Label('b', 4, 8)]

        segment = result[4]
        assert segment.start == 7
        assert segment.end == 8
        assert segment.ref == [annotations.Label('c', 7, 10)]
        assert segment.hyp == [annotations.Label('b', 4, 8)]

        segment = result[5]
        assert segment.start == 8
        assert segment.end == 10
        assert segment.ref == [annotations.Label('c', 7, 10)]
        assert segment.hyp == [annotations.Label('c', 8, 10)]

    def test_align_with_overlapping_labels(self):
        ref = annotations.LabelList(labels=[
            annotations.Label('a', 0, 3),
            annotations.Label('b', 3, 6),
            annotations.Label('bx', 4, 5),
            annotations.Label('c', 7, 10)
        ])

        hyp = annotations.LabelList(labels=[
            annotations.Label('a', 0, 3),
            annotations.Label('b', 4, 8),
            annotations.Label('c', 8, 10),
            annotations.Label('cx', 9, 11)
        ])

        result = alignment.InvariantSegmentAligner().align(ref, hyp)

        assert len(result) == 9

        segment = result[0]
        assert segment.start == 0
        assert segment.end == 3
        assert segment.ref == [annotations.Label('a', 0, 3)]
        assert segment.hyp == [annotations.Label('a', 0, 3)]

        segment = result[1]
        assert segment.start == 3
        assert segment.end == 4
        assert segment.ref == [annotations.Label('b', 3, 6)]
        assert segment.hyp == []

        segment = result[2]
        assert segment.start == 4
        assert segment.end == 5
        assert segment.ref == [annotations.Label('b', 3, 6), annotations.Label('bx', 4, 5)]
        assert segment.hyp == [annotations.Label('b', 4, 8)]

        segment = result[3]
        assert segment.start == 5
        assert segment.end == 6
        assert segment.ref == [annotations.Label('b', 3, 6)]
        assert segment.hyp == [annotations.Label('b', 4, 8)]

        segment = result[4]
        assert segment.start == 6
        assert segment.end == 7
        assert segment.ref == []
        assert segment.hyp == [annotations.Label('b', 4, 8)]

        segment = result[5]
        assert segment.start == 7
        assert segment.end == 8
        assert segment.ref == [annotations.Label('c', 7, 10)]
        assert segment.hyp == [annotations.Label('b', 4, 8)]

        segment = result[6]
        assert segment.start == 8
        assert segment.end == 9
        assert segment.ref == [annotations.Label('c', 7, 10)]
        assert segment.hyp == [annotations.Label('c', 8, 10)]

        segment = result[7]
        assert segment.start == 9
        assert segment.end == 10
        assert segment.ref == [annotations.Label('c', 7, 10)]
        assert segment.hyp == [annotations.Label('c', 8, 10), annotations.Label('cx', 9, 11)]

        segment = result[8]
        assert segment.start == 10
        assert segment.end == 11
        assert segment.ref == []
        assert segment.hyp == [annotations.Label('cx', 9, 11)]

    def test_align_with_empty_segments(self):
        ref = annotations.LabelList(labels=[
            annotations.Label('a', 0, 3),
            annotations.Label('b', 4, 6),
        ])

        hyp = annotations.LabelList(labels=[
            annotations.Label('a', 0, 3),
            annotations.Label('c', 5, 8),
        ])

        result = alignment.InvariantSegmentAligner().align(ref, hyp)

        assert len(result) == 5

        segment = result[0]
        assert segment.start == 0
        assert segment.end == 3
        assert segment.ref == [annotations.Label('a', 0, 3)]
        assert segment.hyp == [annotations.Label('a', 0, 3)]

        segment = result[1]
        assert segment.start == 3
        assert segment.end == 4
        assert segment.ref == []
        assert segment.hyp == []

        segment = result[2]
        assert segment.start == 4
        assert segment.end == 5
        assert segment.ref == [annotations.Label('b', 4, 6)]
        assert segment.hyp == []

        segment = result[3]
        assert segment.start == 5
        assert segment.end == 6
        assert segment.ref == [annotations.Label('b', 4, 6)]
        assert segment.hyp == [annotations.Label('c', 5, 8)]

        segment = result[4]
        assert segment.start == 6
        assert segment.end == 8
        assert segment.ref == []
        assert segment.hyp == [annotations.Label('c', 5, 8)]

    def test_align_empty_ground_truth(self):
        ref = annotations.LabelList(labels=[
        ])

        hyp = annotations.LabelList(labels=[
            annotations.Label('b', 4, 8)
        ])

        result = alignment.InvariantSegmentAligner().align(ref, hyp)

        assert len(result) == 1

        segment = result[0]
        assert segment.start == 4
        assert segment.end == 8
        assert segment.ref == []
        assert segment.hyp == [annotations.Label('b', 4, 8)]

    def test_align_empty_hypothesis(self):
        ref = annotations.LabelList(labels=[
            annotations.Label('b', 4, 8)
        ])

        hyp = annotations.LabelList(labels=[
        ])

        result = alignment.InvariantSegmentAligner().align(ref, hyp)

        assert len(result) == 1

        segment = result[0]
        assert segment.start == 4
        assert segment.end == 8
        assert segment.ref == [annotations.Label('b', 4, 8)]
        assert segment.hyp == []

    def test_align_different_start(self):
        ref = annotations.LabelList(labels=[
            annotations.Label('b', 4, 9)
        ])

        hyp = annotations.LabelList(labels=[
            annotations.Label('b', 2, 8)
        ])

        result = alignment.InvariantSegmentAligner().align(ref, hyp)

        assert len(result) == 3

        segment = result[0]
        assert segment.start == 2
        assert segment.end == 4
        assert segment.ref == []
        assert segment.hyp == [annotations.Label('b', 2, 8)]

        segment = result[1]
        assert segment.start == 4
        assert segment.end == 8
        assert segment.ref == [annotations.Label('b', 4, 9)]
        assert segment.hyp == [annotations.Label('b', 2, 8)]

        segment = result[2]
        assert segment.start == 8
        assert segment.end == 9
        assert segment.ref == [annotations.Label('b', 4, 9)]
        assert segment.hyp == []

    def test_create_event_list(self):
        ll_ref = annotations.LabelList(labels=[
            annotations.Label('a', 0.89, 13.73),
            annotations.Label('a', 13.73, 17.49),
            annotations.Label('b', 17.49, 22.75)
        ])

        ll_hyp = annotations.LabelList(labels=[
            annotations.Label('b', 0.1, 1.656),
            annotations.Label('a', 1.656, 1.976),
            annotations.Label('b', 1.976, 3.896),
            annotations.Label('a', 3.896, 3.957)
        ])

        events = alignment.InvariantSegmentAligner.create_event_list(ll_ref, ll_hyp, time_threshold=0.01)

        assert events[0] == (0.1, [(0.1, 'S', 1, annotations.Label('b', 0.1, 1.656))])
        assert events[1] == (0.89, [(0.89, 'S', 0, annotations.Label('a', 0.89, 13.73))])
        assert events[2] == (1.656, [(1.656, 'E', 1, annotations.Label('b', 0.1, 1.656)),
                                     (1.656, 'S', 1, annotations.Label('a', 1.656, 1.976))])
        assert events[3] == (1.976, [(1.976, 'E', 1, annotations.Label('a', 1.656, 1.976)),
                                     (1.976, 'S', 1, annotations.Label('b', 1.976, 3.896))])
        assert events[4] == (3.896, [(3.896, 'E', 1, annotations.Label('b', 1.976, 3.896)),
                                     (3.896, 'S', 1, annotations.Label('a', 3.896, 3.957))])
        assert events[5] == (3.957, [(3.957, 'E', 1, annotations.Label('a', 3.896, 3.957))])
        assert events[6] == (13.73, [(13.73, 'E', 0, annotations.Label('a', 0.89, 13.73)),
                                     (13.73, 'S', 0, annotations.Label('a', 13.73, 17.49))])
        assert events[7] == (17.49, [(17.49, 'E', 0, annotations.Label('a', 13.73, 17.49)),
                                     (17.49, 'S', 0, annotations.Label('b', 17.49, 22.75))])
        assert events[8] == (22.75, [(22.75, 'E', 0, annotations.Label('b', 17.49, 22.75))])
