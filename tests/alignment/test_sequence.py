from audiomate.corpus import assets

from evalmate import alignment


def ll_with_values(values):
    ll = assets.LabelList()

    for value in values:
        ll.append(assets.Label(value))

    return ll


class TestLevenshteinAligner:

    def test_align_deletion(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'c'])
        )

        assert ali == [
            alignment.LabelPair(assets.Label('a'), assets.Label('a')),
            alignment.LabelPair(assets.Label('b'), None),
            alignment.LabelPair(assets.Label('c'), assets.Label('c')),
        ]

    def test_align_insertion(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'b', 'd', 'c'])
        )

        assert ali == [
            alignment.LabelPair(assets.Label('a'), assets.Label('a')),
            alignment.LabelPair(assets.Label('b'), assets.Label('b')),
            alignment.LabelPair(None, assets.Label('b')),
            alignment.LabelPair(assets.Label('c'), assets.Label('c')),
        ]

    def test_align_substitution(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'x', 'c'])
        )

        assert ali == [
            alignment.LabelPair(assets.Label('a'), assets.Label('a')),
            alignment.LabelPair(assets.Label('b'), assets.Label('x')),
            alignment.LabelPair(assets.Label('c'), assets.Label('c')),
        ]

    def test_align_empty_hyp_returns_all_none(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values([])
        )

        assert ali == [
            alignment.LabelPair(assets.Label('a'), None),
            alignment.LabelPair(assets.Label('b'), None),
            alignment.LabelPair(assets.Label('c'), None),
        ]

    def test_align_empty_ref_returns_all_none(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values([]),
            ll_with_values(['a', 'b', 'c'])
        )

        assert ali == [
            alignment.LabelPair(None, assets.Label('a')),
            alignment.LabelPair(None, assets.Label('b')),
            alignment.LabelPair(None, assets.Label('c')),
        ]

    def test_align_high_substitution_cost_forces_deletions_and_insertions(self):
        lev = alignment.LevenshteinAligner(substitution_cost=20)

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'x', 'c'])
        )

        assert ali == [
            alignment.LabelPair(assets.Label('a'), assets.Label('a')),
            alignment.LabelPair(assets.Label('b'), None),
            alignment.LabelPair(None, assets.Label('x')),
            alignment.LabelPair(assets.Label('c'), assets.Label('c')),
        ]
