from audiomate.corpus import assets

from evalmate import alignment
from evalmate import evaluator


class TestASREvaluator:

    def test_evaluate(self):
        ref = evaluator.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[assets.Label('a b a d f a b')])
            }
        )

        hyp = evaluator.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[assets.Label('a b d f i b')])
            }
        )

        result = evaluator.ASREvaluator().do_evaluate(ref, hyp)

        assert len(result.utt_to_label_pairs) == 1
        assert result.utt_to_label_pairs['a'] == [
            alignment.LabelPair(assets.Label('a'), assets.Label('a')),
            alignment.LabelPair(assets.Label('b'), assets.Label('b')),
            alignment.LabelPair(assets.Label('a'), None),
            alignment.LabelPair(assets.Label('d'), assets.Label('d')),
            alignment.LabelPair(assets.Label('f'), assets.Label('f')),
            alignment.LabelPair(assets.Label('a'), assets.Label('i')),
            alignment.LabelPair(assets.Label('b'), assets.Label('b')),
        ]

    def test_evaluate_with_multiple_labels(self):
        ref = evaluator.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[
                    assets.Label('a b', start=0, end=3),
                    assets.Label('a d', start=3, end=5),
                    assets.Label('f a b', start=5, end=6)
                ])
            }
        )

        hyp = evaluator.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[assets.Label('a b d f i b')])
            }
        )

        result = evaluator.ASREvaluator().do_evaluate(ref, hyp)

        assert len(result.utt_to_label_pairs) == 1
        assert result.utt_to_label_pairs['a'] == [
            alignment.LabelPair(assets.Label('a'), assets.Label('a')),
            alignment.LabelPair(assets.Label('b'), assets.Label('b')),
            alignment.LabelPair(assets.Label('a'), None),
            alignment.LabelPair(assets.Label('d'), assets.Label('d')),
            alignment.LabelPair(assets.Label('f'), assets.Label('f')),
            alignment.LabelPair(assets.Label('a'), assets.Label('i')),
            alignment.LabelPair(assets.Label('b'), assets.Label('b')),
        ]
