from audiomate.corpus import assets

from evalmate import tasks


class TestASREvaluator:

    def test_evaluate(self):
        ref = tasks.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[assets.Label('a b a d f a b')])
            }
        )

        hyp = tasks.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[assets.Label('a b d f i b')])
            }
        )

        result = tasks.ASREvaluator().do_evaluate(ref, hyp)

        assert result.aligned_tokens == {
            'a': (['a', 'b', 'a', 'd', 'f', 'a', 'b'],
                  ['a', 'b', None, 'd', 'f', 'i', 'b'])
        }

    def test_evaluate_with_multiple_labels(self):
        ref = tasks.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[
                    assets.Label('a b', start=0, end=3),
                    assets.Label('a d', start=3, end=5),
                    assets.Label('f a b', start=5, end=6)
                ])
            }
        )

        hyp = tasks.Outcome(
            label_lists={
                'a': assets.LabelList(labels=[assets.Label('a b d f i b')])
            }
        )

        result = tasks.ASREvaluator().do_evaluate(ref, hyp)

        assert result.aligned_tokens == {
            'a': (['a', 'b', 'a', 'd', 'f', 'a', 'b'],
                  ['a', 'b', None, 'd', 'f', 'i', 'b'])
        }
