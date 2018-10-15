from audiomate.corpus import assets

from evalmate.utils import structure

import pytest


class TestLabelPair:

    def test_ordering(self):
        pair_a = structure.LabelPair(assets.Label('a1', start=1.55, end=1.88),
                                     assets.Label('a2', start=1.66, end=1.92))

        pair_b = structure.LabelPair(assets.Label('b1', start=1.59, end=1.88),
                                     assets.Label('b2', start=1.66, end=1.92))

        assert pair_a < pair_b

    def test_ordering_equal(self):
        pair_a = structure.LabelPair(assets.Label('a1', start=1.55, end=1.88),
                                     assets.Label('a2', start=1.66, end=1.92))

        pair_b = structure.LabelPair(assets.Label('a1', start=1.55, end=1.88),
                                     assets.Label('a2', start=1.66, end=1.92))

        assert pair_a == pair_b

    def test_ordering_ref_none(self):
        pair_a = structure.LabelPair(None,
                                     assets.Label('a2', start=1.66, end=1.92))

        pair_b = structure.LabelPair(assets.Label('b1', start=1.59, end=1.88),
                                     assets.Label('b2', start=1.66, end=1.92))

        assert pair_b < pair_a

    def test_ordering_both_ref_none(self):
        pair_a = structure.LabelPair(None,
                                     assets.Label('a2', start=1.66, end=1.92))

        pair_b = structure.LabelPair(None,
                                     assets.Label('b2', start=1.65, end=1.92))

        assert pair_b < pair_a

    def test_ordering_only_end_differs(self):
        pair_a = structure.LabelPair(assets.Label('a1', start=1.55, end=1.88),
                                     assets.Label('a2', start=1.66, end=1.92))

        pair_b = structure.LabelPair(assets.Label('b1', start=1.55, end=1.87),
                                     assets.Label('b2', start=1.66, end=1.92))

        assert pair_b < pair_a

    def test_ordering_only_value_differs(self):
        pair_a = structure.LabelPair(assets.Label('a1', start=1.55, end=1.88),
                                     assets.Label('a2', start=1.66, end=1.92))

        pair_b = structure.LabelPair(assets.Label('b1', start=1.55, end=1.88),
                                     assets.Label('b2', start=1.66, end=1.92))

        assert pair_a < pair_b


class TestSegment:

    def test_duration(self):
        seg = structure.Segment(0.8, 1.9)

        assert seg.duration == pytest.approx(1.1)

    def test_compare_single_labels_with_same_times_returns_smaller_label_value(self):
        seg_a = structure.Segment(0.0, 1.0, ref=assets.Label('a'))
        seg_b = structure.Segment(0.0, 1.0, ref=assets.Label('b'))

        assert seg_a < seg_b

    def test_compare_single_labels_returns_smaller_end_time(self):
        seg_a = structure.Segment(0.0, 0.8, ref=assets.Label('a'))
        seg_b = structure.Segment(0.0, 0.9, ref=assets.Label('a'))

        assert seg_a < seg_b

    def test_compare_single_labels_returns_smaller_start_time(self):
        seg_a = structure.Segment(0.0, 0.9, ref=assets.Label('a'))
        seg_b = structure.Segment(0.2, 0.7, ref=assets.Label('a'))

        assert seg_a < seg_b

    def test_compare_multi_labels_with_same_times_returns_smaller_label_value(self):
        seg_a = structure.Segment(0.0, 1.0, ref=[
            assets.Label('a'),
            assets.Label('a')
        ])
        seg_b = structure.Segment(0.0, 1.0, ref=[
            assets.Label('a'),
            assets.Label('b')
        ])

        assert seg_a < seg_b

    def test_compare_multi_labels_returns_smaller_end_time(self):
        seg_a = structure.Segment(0.0, 0.9, ref=[
            assets.Label('a'),
            assets.Label('a')
        ])
        seg_b = structure.Segment(0.0, 1.0, ref=[
            assets.Label('a'),
            assets.Label('b')
        ])

        assert seg_a < seg_b

    def test_compare_multi_labels_returns_smaller_start_time(self):
        seg_a = structure.Segment(0.0, 0.9, ref=[
            assets.Label('a'),
            assets.Label('a')
        ])
        seg_b = structure.Segment(0.2, 0.8, ref=[
            assets.Label('a'),
            assets.Label('a')
        ])

        assert seg_a < seg_b

    def test_equals(self):
        seg_a = structure.Segment(0.0, 0.9, ref=[
            assets.Label('a'),
            assets.Label('a')
        ])
        seg_b = structure.Segment(0.0, 0.9, ref=[
            assets.Label('a'),
            assets.Label('a')
        ])

        assert seg_a == seg_b
