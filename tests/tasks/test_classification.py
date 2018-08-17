from audiomate.corpus import assets

from evalmate import tasks
from evalmate.utils import structure

import pytest


class TestClassificationEvaluator:

    def test_evaluate_with_two_label_lists(self, classification_ref_and_hyp_label_list):
        ll_ref, ll_hyp = classification_ref_and_hyp_label_list

        result = tasks.ClassificationEvaluator().evaluate(ll_ref, ll_hyp)

        expected_segments = [
            structure.Segment(0, 4, assets.Label('music', start=0, end=5), assets.Label('music', start=0, end=4)),
            structure.Segment(4, 5, assets.Label('music', start=0, end=5), assets.Label('speech', start=4, end=6)),
            structure.Segment(5, 6, assets.Label('speech', start=5, end=11), assets.Label('speech', start=4, end=6)),
            structure.Segment(6, 8, assets.Label('speech', start=5, end=11), None),
            structure.Segment(8, 11, assets.Label('speech', start=5, end=11), assets.Label('mix', start=8, end=16)),
            structure.Segment(11, 14, assets.Label('mix', start=11, end=14), assets.Label('mix', start=8, end=16)),
            structure.Segment(14, 16, assets.Label('speech', start=14, end=19), assets.Label('mix', start=8, end=16)),
            structure.Segment(16, 19, assets.Label('speech', start=14, end=19),
                              assets.Label('speech', start=16, end=21)),
            structure.Segment(19, 21, None, assets.Label('speech', start=16, end=21))
        ]

        assert isinstance(result, tasks.ClassificationEvaluation)
        assert sorted(result.aligned_segments) == sorted(expected_segments)

    def test_evaluate_corpus_with_hyp_labels(self, classification_ref_corpus_and_hyp_labels):
        ref_corpus, hyps = classification_ref_corpus_and_hyp_labels
        result = tasks.ClassificationEvaluator().evaluate(ref_corpus, hyps)

        assert isinstance(result, tasks.ClassificationEvaluation)

        assert result.confusion.correct == pytest.approx(124.9)
        assert result.confusion.insertions == pytest.approx(1.7)
        assert result.confusion.deletions == pytest.approx(8.3)
        assert result.confusion.substitutions == pytest.approx(36.4)
        assert result.confusion.substitutions_out == pytest.approx(36.4)
        assert result.confusion.total == pytest.approx(169.6)
