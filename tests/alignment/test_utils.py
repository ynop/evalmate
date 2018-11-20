from audiomate import annotations

from evalmate import alignment

import pytest


class TestLabelPair:

    def test_ordering(self):
        pair_a = alignment.LabelPair(annotations.Label('a1', start=1.55, end=1.88),
                                     annotations.Label('a2', start=1.66, end=1.92))

        pair_b = alignment.LabelPair(annotations.Label('b1', start=1.59, end=1.88),
                                     annotations.Label('b2', start=1.66, end=1.92))

        assert pair_a < pair_b

    def test_ordering_equal(self):
        pair_a = alignment.LabelPair(annotations.Label('a1', start=1.55, end=1.88),
                                     annotations.Label('a2', start=1.66, end=1.92))

        pair_b = alignment.LabelPair(annotations.Label('a1', start=1.55, end=1.88),
                                     annotations.Label('a2', start=1.66, end=1.92))

        assert pair_a == pair_b

    def test_ordering_ref_none(self):
        pair_a = alignment.LabelPair(None,
                                     annotations.Label('a2', start=1.66, end=1.92))

        pair_b = alignment.LabelPair(annotations.Label('b1', start=1.59, end=1.88),
                                     annotations.Label('b2', start=1.66, end=1.92))

        assert pair_b < pair_a

    def test_ordering_both_ref_none(self):
        pair_a = alignment.LabelPair(None,
                                     annotations.Label('a2', start=1.66, end=1.92))

        pair_b = alignment.LabelPair(None,
                                     annotations.Label('b2', start=1.65, end=1.92))

        assert pair_b < pair_a

    def test_ordering_only_end_differs(self):
        pair_a = alignment.LabelPair(annotations.Label('a1', start=1.55, end=1.88),
                                     annotations.Label('a2', start=1.66, end=1.92))

        pair_b = alignment.LabelPair(annotations.Label('b1', start=1.55, end=1.87),
                                     annotations.Label('b2', start=1.66, end=1.92))

        assert pair_b < pair_a

    def test_ordering_only_value_differs(self):
        pair_a = alignment.LabelPair(annotations.Label('a1', start=1.55, end=1.88),
                                     annotations.Label('a2', start=1.66, end=1.92))

        pair_b = alignment.LabelPair(annotations.Label('b1', start=1.55, end=1.88),
                                     annotations.Label('b2', start=1.66, end=1.92))

        assert pair_a < pair_b


class TestSegment:

    def test_duration(self):
        seg = alignment.Segment(0.8, 1.9)

        assert seg.duration == pytest.approx(1.1)

    def test_compare_single_labels_with_same_times_returns_smaller_label_value(self):
        seg_a = alignment.Segment(0.0, 1.0, ref=annotations.Label('a'))
        seg_b = alignment.Segment(0.0, 1.0, ref=annotations.Label('b'))

        assert seg_a < seg_b

    def test_compare_single_labels_returns_smaller_end_time(self):
        seg_a = alignment.Segment(0.0, 0.8, ref=annotations.Label('a'))
        seg_b = alignment.Segment(0.0, 0.9, ref=annotations.Label('a'))

        assert seg_a < seg_b

    def test_compare_single_labels_returns_smaller_start_time(self):
        seg_a = alignment.Segment(0.0, 0.9, ref=annotations.Label('a'))
        seg_b = alignment.Segment(0.2, 0.7, ref=annotations.Label('a'))

        assert seg_a < seg_b

    def test_compare_multi_labels_with_same_times_returns_smaller_label_value(self):
        seg_a = alignment.Segment(0.0, 1.0, ref=[
            annotations.Label('a'),
            annotations.Label('a')
        ])
        seg_b = alignment.Segment(0.0, 1.0, ref=[
            annotations.Label('a'),
            annotations.Label('b')
        ])

        assert seg_a < seg_b

    def test_compare_multi_labels_returns_smaller_end_time(self):
        seg_a = alignment.Segment(0.0, 0.9, ref=[
            annotations.Label('a'),
            annotations.Label('a')
        ])
        seg_b = alignment.Segment(0.0, 1.0, ref=[
            annotations.Label('a'),
            annotations.Label('b')
        ])

        assert seg_a < seg_b

    def test_compare_multi_labels_returns_smaller_start_time(self):
        seg_a = alignment.Segment(0.0, 0.9, ref=[
            annotations.Label('a'),
            annotations.Label('a')
        ])
        seg_b = alignment.Segment(0.2, 0.8, ref=[
            annotations.Label('a'),
            annotations.Label('a')
        ])

        assert seg_a < seg_b

    def test_equals(self):
        seg_a = alignment.Segment(0.0, 0.9, ref=[
            annotations.Label('a'),
            annotations.Label('a')
        ])
        seg_b = alignment.Segment(0.0, 0.9, ref=[
            annotations.Label('a'),
            annotations.Label('a')
        ])

        assert seg_a == seg_b
