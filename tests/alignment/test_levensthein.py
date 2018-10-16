from evalmate import alignment


class TestLevenshtein:

    def test_align_deletion(self):
        lev = alignment.LevenshteinAligner()

        ref_ali, hyp_ali = lev.align(['a', 'b', 'c'], ['a', 'c'])

        assert ref_ali == ['a', 'b', 'c']
        assert hyp_ali == ['a', None, 'c']

    def test_align_insertion(self):
        lev = alignment.LevenshteinAligner()

        ref_ali, hyp_ali = lev.align(['a', 'b', 'c'], ['a', 'b', 'd', 'c'])

        assert ref_ali == ['a', 'b', None, 'c']
        assert hyp_ali == ['a', 'b', 'd', 'c']

    def test_align_substitution(self):
        lev = alignment.LevenshteinAligner()

        ref_ali, hyp_ali = lev.align(['a', 'b', 'c'], ['a', 'x', 'c'])

        assert ref_ali == ['a', 'b', 'c']
        assert hyp_ali == ['a', 'x', 'c']

    def test_align_empty_hyp_returns_all_none(self):
        lev = alignment.LevenshteinAligner()

        ref_ali, hyp_ali = lev.align(['a', 'b', 'c'], [])

        assert ref_ali == ['a', 'b', 'c']
        assert hyp_ali == [None, None, None]

    def test_align_empty_ref_returns_all_none(self):
        lev = alignment.LevenshteinAligner()

        ref_ali, hyp_ali = lev.align([], ['a', 'b', 'c'])

        assert ref_ali == [None, None, None]
        assert hyp_ali == ['a', 'b', 'c']

    def test_align_high_substitution_cost_forces_deletions_and_insertions(self):
        lev = alignment.LevenshteinAligner(substitution_cost=20)

        ref_ali, hyp_ali = lev.align(['a', 'b', 'c'], ['a', 'x', 'c'])

        assert ref_ali == ['a', 'b', None, 'c']
        assert hyp_ali == ['a', None, 'x', 'c']
