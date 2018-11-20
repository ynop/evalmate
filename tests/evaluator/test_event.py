from audiomate import annotations

from evalmate import alignment
from evalmate import evaluator

import pytest


@pytest.fixture
def ll_ref():
    return annotations.LabelList(labels=[
        annotations.Label('up', start=5.28, end=5.99),
        annotations.Label('down', start=10.35, end=11.12),
        annotations.Label('right', start=20.87, end=22.01),
        annotations.Label('up', start=33.00, end=33.4),
        annotations.Label('up', start=33.4, end=33.8),
        annotations.Label('down', start=39.28, end=40.0)
    ])


@pytest.fixture
def ll_hyp():
    return annotations.LabelList(labels=[
        annotations.Label('up', start=5.20, end=5.88),
        annotations.Label('right', start=10.30, end=11.08),
        annotations.Label('up', start=32.00, end=32.5),
        annotations.Label('up', start=34.2, end=34.8),
        annotations.Label('left', start=39.3, end=39.9),
        annotations.Label('down', start=39.27, end=40.01)
    ])


class TestEventEvaluator:

    def test_evaluate_computes_correct_alignment(self, ll_ref, ll_hyp):
        aligner = alignment.BipartiteMatchingAligner()
        result = evaluator.EventEvaluator(aligner).evaluate(ll_ref, ll_hyp)

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

        assert isinstance(result, evaluator.EventEvaluation)
        assert sorted(expected_matches) == sorted(result.label_pairs)

    def test_evaluate_computes_confusion(self, ll_ref, ll_hyp):
        aligner = alignment.BipartiteMatchingAligner()
        result = evaluator.EventEvaluator(aligner).evaluate(ll_ref, ll_hyp)

        assert result.confusion.total == 6
        assert result.confusion.correct == 2
        assert result.confusion.substitutions == 1
        assert result.confusion.deletions == 3
        assert result.confusion.insertions == 3
