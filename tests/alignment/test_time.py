from audiomate.corpus import assets

from evalmate import alignment


class TestBipartiteMatchingAligner:
    def test_align(self, kws_ref_and_hyp_label_list):
        ll_ref, ll_hyp = kws_ref_and_hyp_label_list

        aligner = alignment.BipartiteMatchingAligner(start_delta_threshold=0.5,
                                                     end_delta_threshold=-1,
                                                     substitution_penalty=2,
                                                     non_overlap_penalty_weight=1)

        matches = aligner.align(ll_ref, ll_hyp)

        expected_matches = [
            alignment.LabelPair(assets.Label('up', start=5.28, end=5.99),
                                assets.Label('up', start=5.20, end=5.88)),
            alignment.LabelPair(assets.Label('down', start=10.35, end=11.12),
                                assets.Label('right', start=10.30, end=11.08)),
            alignment.LabelPair(assets.Label('right', start=20.87, end=22.01), None),
            alignment.LabelPair(assets.Label('up', start=33.00, end=33.4), None),
            alignment.LabelPair(assets.Label('up', start=33.4, end=33.8), None),
            alignment.LabelPair(assets.Label('down', start=39.28, end=40.0),
                                assets.Label('down', start=39.27, end=40.01)),
            alignment.LabelPair(None, assets.Label('up', start=32.00, end=32.5)),
            alignment.LabelPair(None, assets.Label('up', start=34.2, end=34.8)),
            alignment.LabelPair(None, assets.Label('left', start=39.3, end=39.9))
        ]

        assert sorted(expected_matches) == sorted(matches)

    def test_align_empty_hyp_and_ref_returns_empty_list(self):
        ll_ref = assets.LabelList(labels=[
        ])

        ll_hyp = assets.LabelList(labels=[
        ])

        aligner = alignment.BipartiteMatchingAligner(start_delta_threshold=0.5,
                                                     end_delta_threshold=-1,
                                                     substitution_penalty=2,
                                                     non_overlap_penalty_weight=1)

        matches = aligner.align(ll_ref, ll_hyp)

        assert matches == []

    def test_align_empty_hyp_returns_deletions(self):
        ll_ref = assets.LabelList(labels=[
            assets.Label('greasy', 1.4, 1.9)
        ])

        ll_hyp = assets.LabelList(labels=[
        ])

        aligner = alignment.BipartiteMatchingAligner(start_delta_threshold=0.5,
                                                     end_delta_threshold=-1,
                                                     substitution_penalty=2,
                                                     non_overlap_penalty_weight=1)

        matches = aligner.align(ll_ref, ll_hyp)

        assert matches == [
            alignment.LabelPair(assets.Label('greasy', 1.4, 1.9), None)
        ]

    def test_align_empty_ref_returns_insertions(self):
        ll_ref = assets.LabelList(labels=[
        ])

        ll_hyp = assets.LabelList(labels=[
            assets.Label('greasy', 1.4, 1.9)
        ])

        aligner = alignment.BipartiteMatchingAligner(start_delta_threshold=0.5,
                                                     end_delta_threshold=-1,
                                                     substitution_penalty=2,
                                                     non_overlap_penalty_weight=1)

        matches = aligner.align(ll_ref, ll_hyp)

        assert matches == [
            alignment.LabelPair(None, assets.Label('greasy', 1.4, 1.9))
        ]


class TestFullMatchingAligner:

    def test_align(self):
        ref_ll = assets.LabelList(labels=[
            assets.Label('a', 4.2, 8.5),
            assets.Label('b', 13.1, 19.23)
        ])

        hyp_ll = assets.LabelList(labels=[
            assets.Label('x', 3.2, 5.4),
            assets.Label('y', 7.6, 15.2)
        ])

        result = alignment.FullMatchingAligner(0.1).align(ref_ll, hyp_ll)

        assert result == [
            alignment.LabelPair(assets.Label('a', 4.2, 8.5), assets.Label('x', 3.2, 5.4)),
            alignment.LabelPair(assets.Label('a', 4.2, 8.5), assets.Label('y', 7.6, 15.2)),
            alignment.LabelPair(assets.Label('b', 13.1, 19.23), assets.Label('y', 7.6, 15.2))
        ]

    def test_align_insertion(self):
        ref_ll = assets.LabelList(labels=[
        ])

        hyp_ll = assets.LabelList(labels=[
            assets.Label('y', 7.6, 15.2)
        ])

        result = alignment.FullMatchingAligner(0.1).align(ref_ll, hyp_ll)

        assert result == [
            alignment.LabelPair(None, assets.Label('y', 7.6, 15.2))
        ]

    def test_align_deletion(self):
        ref_ll = assets.LabelList(labels=[
            assets.Label('a', 4.2, 8.5)
        ])

        hyp_ll = assets.LabelList(labels=[
        ])

        result = alignment.FullMatchingAligner(0.1).align(ref_ll, hyp_ll)

        assert result == [
            alignment.LabelPair(assets.Label('a', 4.2, 8.5), None)
        ]

    def test_align_endless_labels(self):
        ref_ll = assets.LabelList(labels=[
            assets.Label('a', 4.2, 8.5),
            assets.Label('b', 13.1, 19.23)
        ])

        hyp_ll = assets.LabelList(labels=[
            assets.Label('x', 9.2, -1)
        ])

        result = alignment.FullMatchingAligner(0.1).align(ref_ll, hyp_ll)

        assert sorted(result) == sorted([
            alignment.LabelPair(assets.Label('a', 4.2, 8.5), None),
            alignment.LabelPair(assets.Label('b', 13.1, 19.23), assets.Label('x', 9.2, -1))
        ])
