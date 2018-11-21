from audiomate import annotations

from evalmate import alignment
from evalmate import confusion

import pytest


@pytest.fixture()
def sample_confusion():
    conf = confusion.EventConfusion('up')

    conf.correct_pairs.extend([
        alignment.LabelPair(annotations.Label('up', start=5.28, end=5.99),
                            annotations.Label('up', start=5.20, end=5.88)),
        alignment.LabelPair(annotations.Label('up', start=39.28, end=40.0),
                            annotations.Label('down', start=39.27, end=40.01))
    ])

    conf.insertion_pairs.extend([
        alignment.LabelPair(None, annotations.Label('up', start=32.00, end=32.5)),
        alignment.LabelPair(None, annotations.Label('up', start=34.2, end=34.8)),
        alignment.LabelPair(None, annotations.Label('up', start=55.3, end=56.9))
    ])

    conf.deletion_pairs.extend([
        alignment.LabelPair(annotations.Label('up', start=20.87, end=22.01), None),
        alignment.LabelPair(annotations.Label('up', start=33.00, end=33.4), None),
        alignment.LabelPair(annotations.Label('up', start=33.4, end=33.8), None),
    ])

    conf.substitution_pairs['right'].extend([
        alignment.LabelPair(annotations.Label('up', start=10.35, end=11.12),
                            annotations.Label('right', start=10.30, end=11.18)),
    ])

    conf.substitution_pairs['down'].extend([
        alignment.LabelPair(annotations.Label('up', start=74.28, end=75.0),
                            annotations.Label('down', start=74.17, end=75.01))
    ])

    conf.substitution_out_pairs['left'].extend([
        alignment.LabelPair(annotations.Label('left', start=15.35, end=16.12),
                            annotations.Label('up', start=15.4, end=16.18)),
    ])

    conf.substitution_out_pairs['up'].extend([
        alignment.LabelPair(annotations.Label('down', start=84.28, end=85.09),
                            annotations.Label('up', start=84.17, end=85.01))
    ])

    return conf


class TestInstanceConfusionStats:

    def test_correct(self, sample_confusion):
        assert sample_confusion.correct == 2

    def test_insertions(self, sample_confusion):
        assert sample_confusion.insertions == 3

    def test_deletions(self, sample_confusion):
        assert sample_confusion.deletions == 3

    def test_substitutions(self, sample_confusion):
        assert sample_confusion.substitutions == 2

    def test_substitutions_out(self, sample_confusion):
        assert sample_confusion.substitutions_out == 2

    def test_total(self, sample_confusion):
        assert sample_confusion.total == 7

    def test_substitutions_by_count(self, sample_confusion):
        assert sample_confusion.substitutions_by_count() == [
            ('down', 1),
            ('right', 1)
        ]
