from audiomate import annotations

from evalmate import alignment
from evalmate import evaluator


class TestASREvaluator:

    def test_evaluate(self):
        ref = evaluator.Outcome(
            label_lists={
                'a': annotations.LabelList(labels=[annotations.Label('a b a d f a b')])
            }
        )

        hyp = evaluator.Outcome(
            label_lists={
                'a': annotations.LabelList(labels=[annotations.Label('a b d f i b')])
            }
        )

        result = evaluator.ASREvaluator().do_evaluate(ref, hyp)

        assert len(result.utt_to_label_pairs) == 1
        assert result.utt_to_label_pairs['a'] == [
            alignment.LabelPair(annotations.Label('a'), annotations.Label('a')),
            alignment.LabelPair(annotations.Label('b'), annotations.Label('b')),
            alignment.LabelPair(annotations.Label('a'), None),
            alignment.LabelPair(annotations.Label('d'), annotations.Label('d')),
            alignment.LabelPair(annotations.Label('f'), annotations.Label('f')),
            alignment.LabelPair(annotations.Label('a'), annotations.Label('i')),
            alignment.LabelPair(annotations.Label('b'), annotations.Label('b')),
        ]

    def test_evaluate_with_multiple_labels(self):
        ref = evaluator.Outcome(
            label_lists={
                'a': annotations.LabelList(labels=[
                    annotations.Label('a b', start=0, end=3),
                    annotations.Label('a d', start=3, end=5),
                    annotations.Label('f a b', start=5, end=6)
                ])
            }
        )

        hyp = evaluator.Outcome(
            label_lists={
                'a': annotations.LabelList(labels=[annotations.Label('a b d f i b')])
            }
        )

        result = evaluator.ASREvaluator().do_evaluate(ref, hyp)

        assert len(result.utt_to_label_pairs) == 1
        assert result.utt_to_label_pairs['a'] == [
            alignment.LabelPair(annotations.Label('a'), annotations.Label('a')),
            alignment.LabelPair(annotations.Label('b'), annotations.Label('b')),
            alignment.LabelPair(annotations.Label('a'), None),
            alignment.LabelPair(annotations.Label('d'), annotations.Label('d')),
            alignment.LabelPair(annotations.Label('f'), annotations.Label('f')),
            alignment.LabelPair(annotations.Label('a'), annotations.Label('i')),
            alignment.LabelPair(annotations.Label('b'), annotations.Label('b')),
        ]
