from audiomate.corpus import assets

from evalmate.utils import structure
from evalmate import tasks


class TestKWSEvaluator:

    def test_evaluate_with_two_label_lists(self, kws_ref_and_hyp_label_list):
        ll_ref, ll_hyp = kws_ref_and_hyp_label_list

        result = tasks.KWSEvaluator().evaluate(ll_ref, ll_hyp)

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

        assert isinstance(result, tasks.KWSEvaluation)
        assert sorted(expected_matches) == sorted(result.aligned_labels)

    def test_evaluate_corpus_with_hyp_labels(self, kws_ref_corpus_and_hyp_labels):
        result = tasks.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        assert isinstance(result, tasks.KWSEvaluation)

        assert result.confusion_stats.total == 24
        assert result.confusion_stats.correct == 17
        assert result.confusion_stats.substitutions == 4
        assert result.confusion_stats.deletions == 3
        assert result.confusion_stats.insertions == 6
