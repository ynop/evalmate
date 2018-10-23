from evalmate import alignment
from evalmate import confusion

import pytest


@pytest.fixture
def sample_confusion():
    cnf = confusion.SegmentConfusion('music')

    cnf.correct_segments = [
        alignment.Segment(0, 3),
        alignment.Segment(88, 103),
        alignment.Segment(159.2, 193.1)
    ]

    cnf.insertion_segments = [
        alignment.Segment(8, 19),
        alignment.Segment(55, 67.2)
    ]

    cnf.deletion_segments = [
        alignment.Segment(22, 28.2),
        alignment.Segment(31, 38.2),
        alignment.Segment(115, 124.9)
    ]

    cnf.substitution_segments = {
        'speech': [
            alignment.Segment(44, 48.9),
            alignment.Segment(198, 204.9)
        ],
        'mix': [
            alignment.Segment(133.4, 141.2)
        ]
    }

    cnf.substitution_out_segments = {
        'speech': [
            alignment.Segment(208.9, 210.2)
        ],
        'mix': [
            alignment.Segment(70.2, 79.78),
            alignment.Segment(4, 7)
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
