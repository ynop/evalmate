from audiomate import annotations
import numpy as np

from evalmate import alignment
from evalmate import evaluator

import pytest


class TestKWSEvaluator:

    def test_evaluate_with_two_label_lists(self, kws_ref_and_hyp_label_list):
        ll_ref, ll_hyp = kws_ref_and_hyp_label_list

        result = evaluator.KWSEvaluator().evaluate(ll_ref, ll_hyp)

        expected_matches = [
            alignment.LabelPair(annotations.Label('up', start=5.28, end=5.99),
                                annotations.Label('up', start=5.20, end=5.88)),
            alignment.LabelPair(annotations.Label('down', start=10.35, end=11.12),
                                annotations.Label('right', start=10.30, end=11.08)),
            alignment.LabelPair(annotations.Label('right', start=20.87, end=22.01), None),
            alignment.LabelPair(annotations.Label('up', start=33.00, end=33.4), None),
            alignment.LabelPair(annotations.Label('up', start=33.4, end=33.8), None),
            alignment.LabelPair(annotations.Label('down', start=39.28, end=40.0),
                                annotations.Label('down', start=39.27, end=40.01)),
            alignment.LabelPair(None, annotations.Label('up', start=32.00, end=32.5)),
            alignment.LabelPair(None, annotations.Label('up', start=34.2, end=34.8)),
            alignment.LabelPair(None, annotations.Label('left', start=39.3, end=39.9))
        ]

        assert isinstance(result, evaluator.KWSEvaluation)
        assert sorted(expected_matches) == sorted(result.label_pairs)

    def test_evaluate_corpus_with_hyp_labels(self, kws_ref_corpus_and_hyp_labels):
        result = evaluator.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        assert isinstance(result, evaluator.KWSEvaluation)

        assert result.confusion.total == 24
        assert result.confusion.correct == 17
        assert result.confusion.substitutions == 4
        assert result.confusion.deletions == 3
        assert result.confusion.insertions == 6

    def test_evaluate_with_empty_hyp(self):
        ref = evaluator.Outcome(label_lists={
            'a': annotations.LabelList(labels=[
                annotations.Label('one', 2.5, 4.5),
                annotations.Label('two', 10.5, 11.5),
            ]),
            'b': annotations.LabelList(labels=[
                annotations.Label('one', 1.5, 1.9),
            ]),
            'c': annotations.LabelList(labels=[
                annotations.Label('two', 4.5, 4.9),
                annotations.Label('two', 10.5, 11.5),
            ]),
        })
        hyp = evaluator.Outcome(label_lists={
            'a': annotations.LabelList(labels=[
                annotations.Label('one', 2.5, 4.5),
                annotations.Label('two', 10.5, 11.5),
            ]),
            'b': annotations.LabelList(labels=[
            ]),
            'c': annotations.LabelList(labels=[
                annotations.Label('two', 4.5, 4.9),
                annotations.Label('two', 10.5, 11.5),
            ]),
        })

        result = evaluator.KWSEvaluator().evaluate(ref, hyp)

        assert result.confusion.total == 5


class TestKWSEvaluation:

    def test_false_rejection_rate(self, kws_ref_corpus_and_hyp_labels):
        result = evaluator.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        per_keyword = [2 / 7, 1 / 4, 2 / 6, 1 / 3, 1 / 4]
        assert result.false_rejection_rate() == pytest.approx(np.mean(per_keyword))

    def test_false_rejection_rate_with_no_occurences_returns_zero(self):
        result = evaluator.KWSEvaluator().evaluate(
            annotations.LabelList(labels=[
            ]),
            annotations.LabelList(labels=[
                annotations.Label('four', 2.5, 1.0)
            ])
        )

        assert result.false_rejection_rate() == 0.0

    def test_false_rejection_rate_for_single_keyword(self, kws_ref_corpus_and_hyp_labels):
        result = evaluator.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        assert result.false_rejection_rate(keyword='four') == pytest.approx(1.0 / 4.0)

    def test_false_alarm_rate(self, kws_ref_corpus_and_hyp_labels):
        result = evaluator.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        per_keyword = np.array([0 / 143.4, 4 / 146.4, 2 / 144.4, 1 / 147.4, 3 / 146.4])
        assert result.false_alarm_rate() == pytest.approx(np.mean(per_keyword))

    def test_false_alarm_rate_for_single_keyword(self, kws_ref_corpus_and_hyp_labels):
        result = evaluator.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        assert result.false_alarm_rate(keyword='four') == pytest.approx(3 / (150.4 - 4))

    def test_term_weighted_value(self, kws_ref_corpus_and_hyp_labels):
        result = evaluator.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        p_miss = np.array([2 / 7, 1 / 4, 2 / 6, 1 / 3, 1 / 4]).mean()
        p_fa = np.array([0 / 143.4, 4 / 146.4, 2 / 144.4, 1 / 147.4, 3 / 146.4]).mean()

        beta = (0.1 / 1.0) * (((10 ** -4) ** -1) - 1)
        twv = 1 - (p_miss + beta * p_fa)

        assert result.term_weighted_value() == pytest.approx(twv)

    def test_term_weighted_value_for_single_keyword(self, kws_ref_corpus_and_hyp_labels):
        result = evaluator.KWSEvaluator().evaluate(kws_ref_corpus_and_hyp_labels[0], kws_ref_corpus_and_hyp_labels[1])

        p_miss = 1.0 / 4.0
        p_fa = 3 / (150.4 - 4)

        beta = (0.1 / 1.0) * (((10 ** -4) ** -1) - 1)
        twv = 1 - (p_miss + beta * p_fa)

        assert result.term_weighted_value(keyword='four') == pytest.approx(twv)
