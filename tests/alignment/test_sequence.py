from audiomate import annotations

from evalmate import alignment


def ll_with_values(values):
    ll = annotations.LabelList()

    for value in values:
        ll.append(annotations.Label(value))

    return ll


class TestLevenshteinAligner:

    def test_align_deletion(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'c'])
        )

        assert ali == [
            alignment.LabelPair(annotations.Label('a'), annotations.Label('a')),
            alignment.LabelPair(annotations.Label('b'), None),
            alignment.LabelPair(annotations.Label('c'), annotations.Label('c')),
        ]

    def test_align_insertion(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'b', 'd', 'c'])
        )

        assert ali == [
            alignment.LabelPair(annotations.Label('a'), annotations.Label('a')),
            alignment.LabelPair(annotations.Label('b'), annotations.Label('b')),
            alignment.LabelPair(None, annotations.Label('b')),
            alignment.LabelPair(annotations.Label('c'), annotations.Label('c')),
        ]

    def test_align_substitution(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'x', 'c'])
        )

        assert ali == [
            alignment.LabelPair(annotations.Label('a'), annotations.Label('a')),
            alignment.LabelPair(annotations.Label('b'), annotations.Label('x')),
            alignment.LabelPair(annotations.Label('c'), annotations.Label('c')),
        ]

    def test_align_empty_hyp_returns_all_none(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values([])
        )

        assert ali == [
            alignment.LabelPair(annotations.Label('a'), None),
            alignment.LabelPair(annotations.Label('b'), None),
            alignment.LabelPair(annotations.Label('c'), None),
        ]

    def test_align_empty_ref_returns_all_none(self):
        lev = alignment.LevenshteinAligner()

        ali = lev.align(
            ll_with_values([]),
            ll_with_values(['a', 'b', 'c'])
        )

        assert ali == [
            alignment.LabelPair(None, annotations.Label('a')),
            alignment.LabelPair(None, annotations.Label('b')),
            alignment.LabelPair(None, annotations.Label('c')),
        ]

    def test_align_high_substitution_cost_forces_deletions_and_insertions(self):
        lev = alignment.LevenshteinAligner(substitution_cost=20)

        ali = lev.align(
            ll_with_values(['a', 'b', 'c']),
            ll_with_values(['a', 'x', 'c'])
        )

        assert ali == [
            alignment.LabelPair(annotations.Label('a'), annotations.Label('a')),
            alignment.LabelPair(annotations.Label('b'), None),
            alignment.LabelPair(None, annotations.Label('x')),
            alignment.LabelPair(annotations.Label('c'), annotations.Label('c')),
        ]
