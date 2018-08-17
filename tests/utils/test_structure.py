from audiomate.corpus import assets

from evalmate.utils import structure


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
