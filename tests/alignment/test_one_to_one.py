from audiomate.corpus import assets
from evalmate.alignment import one_to_one
from evalmate.utils import structure


class TestBipartiteMatchingOneToOneAligner:
    def test_align(self, kws_ref_and_hyp_label_list):
        ll_ref, ll_hyp = kws_ref_and_hyp_label_list

        aligner = one_to_one.BipartiteMatchingOneToOneAligner(start_delta_threshold=0.5,
                                                              end_delta_threshold=-1,
                                                              substitution_penalty=2,
                                                              non_overlap_penalty_weight=1)

        matches = aligner.align(ll_ref, ll_hyp)

        expected_matches = [
            structure.LabelPair(assets.Label('up', start=5.28, end=5.99),
                                assets.Label('up', start=5.20, end=5.88)),
            structure.LabelPair(assets.Label('down', start=10.35, end=11.12),
                                assets.Label('right', start=10.30, end=11.08)),
            structure.LabelPair(assets.Label('right', start=20.87, end=22.01), None),
            structure.LabelPair(assets.Label('up', start=33.00, end=33.4), None),
            structure.LabelPair(assets.Label('up', start=33.4, end=33.8), None),
            structure.LabelPair(assets.Label('down', start=39.28, end=40.0),
                                assets.Label('down', start=39.27, end=40.01)),
            structure.LabelPair(None, assets.Label('up', start=32.00, end=32.5)),
            structure.LabelPair(None, assets.Label('up', start=34.2, end=34.8)),
            structure.LabelPair(None, assets.Label('left', start=39.3, end=39.9))
        ]

        assert sorted(expected_matches) == sorted(matches)
