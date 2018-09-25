from audiomate.corpus import assets
import numpy as np

from evalmate.utils import structure
from evalmate import tasks

import pytest


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

        assert result.confusion.total == 24
        assert result.confusion.correct == 17
        assert result.confusion.substitutions == 4
        assert result.confusion.deletions == 3
        assert result.confusion.insertions == 6


class TestKWSEvaluation:

    def test_false_rejection_rate(self, kws_ref_corpus_and_hyp_labels):
        result = tasks.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        per_keyword = [2 / 7, 1 / 4, 2 / 6, 1 / 3, 1 / 4]
        assert result.false_rejection_rate() == pytest.approx(np.mean(per_keyword))

    def test_false_rejection_rate_for_single_keyword(self, kws_ref_corpus_and_hyp_labels):
        result = tasks.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        assert result.false_rejection_rate(keyword='four') == pytest.approx(1.0 / 4.0)

    def test_false_alarm_rate(self, kws_ref_corpus_and_hyp_labels):
        result = tasks.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        per_keyword = np.array([0 / 143.4, 4 / 146.4, 2 / 144.4, 1 / 147.4, 3 / 146.4])
        assert result.false_alarm_rate() == pytest.approx(np.mean(per_keyword))

    def test_false_alarm_rate_for_single_keyword(self, kws_ref_corpus_and_hyp_labels):
        result = tasks.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        assert result.false_alarm_rate(keyword='four') == pytest.approx(3 / (150.4 - 4))

    def test_term_weighted_value(self, kws_ref_corpus_and_hyp_labels):
        result = tasks.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        p_miss = np.array([2 / 7, 1 / 4, 2 / 6, 1 / 3, 1 / 4]).mean()
        p_fa = np.array([0 / 143.4, 4 / 146.4, 2 / 144.4, 1 / 147.4, 3 / 146.4]).mean()

        beta = (0.1 / 1.0) * (((10 ** -4) ** -1) - 1)
        twv = 1 - (p_miss + beta * p_fa)

        assert result.term_weighted_value() == pytest.approx(twv)

    def test_term_weighted_value_for_single_keyword(self, kws_ref_corpus_and_hyp_labels):
        result = tasks.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        p_miss = 1.0 / 4.0
        p_fa = 3 / (150.4 - 4)

        beta = (0.1 / 1.0) * (((10 ** -4) ** -1) - 1)
        twv = 1 - (p_miss + beta * p_fa)

        assert result.term_weighted_value(keyword='four') == pytest.approx(twv)
