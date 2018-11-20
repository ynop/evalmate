from audiomate import annotations

from evalmate import alignment
from evalmate import evaluator

import pytest


class TestSegmentEvaluator:

    def test_evaluate_with_two_label_lists(self, classification_ref_and_hyp_label_list):
        ll_ref, ll_hyp = classification_ref_and_hyp_label_list

        result = evaluator.SegmentEvaluator().evaluate(ll_ref, ll_hyp)

        expected_segments = [
            alignment.Segment(0, 4, annotations.Label('music', start=0, end=5),
                              annotations.Label('music', start=0, end=4)),
            alignment.Segment(4, 5, annotations.Label('music', start=0, end=5),
                              annotations.Label('speech', start=4, end=6)),
            alignment.Segment(5, 6, annotations.Label('speech', start=5, end=11),
                              annotations.Label('speech', start=4, end=6)),
            alignment.Segment(6, 8, annotations.Label('speech', start=5, end=11), None),
            alignment.Segment(8, 11, annotations.Label('speech', start=5, end=11),
                              annotations.Label('mix', start=8, end=16)),
            alignment.Segment(11, 14, annotations.Label('mix', start=11, end=14),
                              annotations.Label('mix', start=8, end=16)),
            alignment.Segment(14, 16, annotations.Label('speech', start=14, end=19),
                              annotations.Label('mix', start=8, end=16)),
            alignment.Segment(16, 19, annotations.Label('speech', start=14, end=19),
                              annotations.Label('speech', start=16, end=21)),
            alignment.Segment(19, 21, None, annotations.Label('speech', start=16, end=21))
        ]

        assert isinstance(result, evaluator.SegmentEvaluation)
        assert sorted(result.utt_to_segments[evaluator.Evaluator.DEFAULT_UTT_IDX]) == sorted(expected_segments)

    def test_evaluate_corpus_with_hyp_labels(self, classification_ref_corpus_and_hyp_labels):
        ref_corpus, hyps = classification_ref_corpus_and_hyp_labels
        result = evaluator.SegmentEvaluator().evaluate(ref_corpus, hyps)

        assert isinstance(result, evaluator.SegmentEvaluation)

        assert result.confusion.correct == pytest.approx(124.9)
        assert result.confusion.insertions == pytest.approx(1.7)
        assert result.confusion.deletions == pytest.approx(8.3)
        assert result.confusion.substitutions == pytest.approx(36.4)
        assert result.confusion.substitutions_out == pytest.approx(36.4)
        assert result.confusion.total == pytest.approx(169.6)

    def test_evaluate_with_invalid_labels_raises_error(self):
        ll_ref = annotations.LabelList(labels=[
            annotations.Label('a', 0.89, 13.73),
            annotations.Label('a', 13.73, 17.49),
            annotations.Label('b', 17.49, 22.75)
        ])

        ll_hyp = annotations.LabelList(labels=[
            annotations.Label('b', 0.89, 13.73),
            annotations.Label('b', 13.73, 17.49),
            annotations.Label('a', 17.49, 17.30)
        ])

        with pytest.raises(ValueError) as excinfo:
            evaluator.SegmentEvaluator().evaluate(ll_ref, ll_hyp)

        assert 'Label-end {} is smaller than label-start {}!'.format(17.30, 17.49) in str(excinfo.value)
