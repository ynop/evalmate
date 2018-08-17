from evalmate.utils import structure
from evalmate import confusion

import pytest


@pytest.fixture
def sample_confusion():
    cnf = confusion.SegmentConfusion('music')

    cnf.correct_segments = [
        structure.Segment(0, 3),
        structure.Segment(88, 103),
        structure.Segment(159.2, 193.1)
    ]

    cnf.insertion_segments = [
        structure.Segment(8, 19),
        structure.Segment(55, 67.2)
    ]

    cnf.deletion_segments = [
        structure.Segment(22, 28.2),
        structure.Segment(31, 38.2),
        structure.Segment(115, 124.9)
    ]

    cnf.substitution_segments = {
        'speech': [
            structure.Segment(44, 48.9),
            structure.Segment(198, 204.9)
        ],
        'mix': [
            structure.Segment(133.4, 141.2)
        ]
    }

    cnf.substitution_out_segments = {
        'speech': [
            structure.Segment(208.9, 210.2)
        ],
        'mix': [
            structure.Segment(70.2, 79.78),
            structure.Segment(4, 7)
        ]
    }

    return cnf


class TestSegmentConfusion:

    def test_correct(self, sample_confusion):
        assert sample_confusion.correct == pytest.approx(51.9)

    def test_insertions(self, sample_confusion):
        assert sample_confusion.insertions == pytest.approx(23.2)

    def test_deletions(self, sample_confusion):
        assert sample_confusion.deletions == pytest.approx(23.3)

    def test_substitutions(self, sample_confusion):
        assert sample_confusion.substitutions == pytest.approx(19.6)

    def test_substitutions_out(self, sample_confusion):
        assert sample_confusion.substitutions_out == pytest.approx(13.88)
