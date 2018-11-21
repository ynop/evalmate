from audiomate import annotations

import pytest

from evalmate.evaluator import outcome


@pytest.fixture
def sample_outcome():
    a = annotations.LabelList(labels=[
        annotations.Label('up', start=5.28, end=5.99),
        annotations.Label('down', start=10.35, end=11.12),
        annotations.Label('right', start=20.87, end=22.01),
        annotations.Label('up', start=33.00, end=33.4),
        annotations.Label('up', start=33.4, end=33.8),
        annotations.Label('down', start=39.28, end=40.0)
    ])

    b = annotations.LabelList(labels=[
        annotations.Label('up', start=1.2, end=4.55),
        annotations.Label('right', start=8.37, end=14.01),
        annotations.Label('down', start=31.20, end=33.4),
        annotations.Label('up', start=33.4, end=33.8),
        annotations.Label('down', start=39.28, end=40.0)
    ])

    c = annotations.LabelList(labels=[
        annotations.Label('up', start=2.98, end=5.92),
        annotations.Label('left', start=9.35, end=13.12),
        annotations.Label('right', start=15.87, end=19.01),
        annotations.Label('down', start=28.20, end=33.4),
        annotations.Label('up', start=33.4, end=33.8),
    ])

    return outcome.Outcome(label_lists={'a': a, 'b': b, 'c': c},
                           utterance_durations={'a': 41.8, 'b': 48.9, 'c': 37.2})


class TestOutcome:

    def test_label_set(self, sample_outcome):
        ls = sample_outcome.label_set()

        expected = [
            annotations.Label('up', start=5.28, end=5.99),
            annotations.Label('down', start=10.35, end=11.12),
            annotations.Label('right', start=20.87, end=22.01),
            annotations.Label('up', start=33.00, end=33.4),
            annotations.Label('up', start=33.4, end=33.8),
            annotations.Label('down', start=39.28, end=40.0),
            annotations.Label('up', start=1.2, end=4.55),
            annotations.Label('right', start=8.37, end=14.01),
            annotations.Label('down', start=31.20, end=33.4),
            annotations.Label('up', start=33.4, end=33.8),
            annotations.Label('down', start=39.28, end=40.0),
            annotations.Label('up', start=2.98, end=5.92),
            annotations.Label('left', start=9.35, end=13.12),
            annotations.Label('right', start=15.87, end=19.01),
            annotations.Label('down', start=28.20, end=33.4),
            annotations.Label('up', start=33.4, end=33.8),
        ]

        assert sorted(expected) == sorted(ls.labels)

    def test_label_set_for_value(self, sample_outcome):
        ls = sample_outcome.label_set_for_value('down')

        expected = [
            annotations.Label('down', start=10.35, end=11.12),
            annotations.Label('down', start=39.28, end=40.0),
            annotations.Label('down', start=31.20, end=33.4),
            annotations.Label('down', start=39.28, end=40.0),
            annotations.Label('down', start=28.20, end=33.4)
        ]

        assert sorted(expected) == sorted(ls.labels)

    def test_total_duration(self, sample_outcome):
        assert sample_outcome.total_duration == pytest.approx(41.8 + 48.9 + 37.2)

    def test_all_values(self, sample_outcome):
        assert sample_outcome.all_values == {'up', 'down', 'left', 'right'}


class TestLabelSet:

    def test_count(self, sample_outcome):
        ls = sample_outcome.label_set_for_value('up')

        assert ls.count == 7

    def test_length_min(self, sample_outcome):
        ls = sample_outcome.label_set_for_value('up')

        assert ls.length_min == pytest.approx(0.4)

    def test_length_max(self, sample_outcome):
        ls = sample_outcome.label_set_for_value('up')

        assert ls.length_max == pytest.approx(3.35)

    def test_length_mean(self, sample_outcome):
        ls = sample_outcome.label_set_for_value('up')

        assert ls.length_mean == pytest.approx((0.71 + 0.4 + 0.4 + 3.35 + 0.4 + 2.94 + 0.4) / 7)

    def test_length_median(self, sample_outcome):
        ls = sample_outcome.label_set_for_value('up')

        assert ls.length_median == pytest.approx(0.4)

    def test_length_variance(self, sample_outcome):
        ls = sample_outcome.label_set_for_value('up')

        assert ls.length_variance == pytest.approx(1.4920693877551021)
