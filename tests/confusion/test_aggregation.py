from evalmate import confusion

import pytest


@pytest.fixture
def sample_confusion():
    class MockInstanceConf(confusion.Confusion):

        def __init__(self, c, i, d, s, so):
            self.c = c
            self.i = i
            self.d = d
            self.s = s
            self.so = so

        @property
        def correct(self):
            return self.c

        @property
        def insertions(self):
            return self.i

        @property
        def deletions(self):
            return self.d

        @property
        def substitutions(self):
            return self.s

        @property
        def substitutions_out(self):
            return self.so

    cnf = confusion.AggregatedConfusion()

    cnf.instances['a'] = MockInstanceConf(10.2, 3.4, 2.9, 4.3, 1.2)
    cnf.instances['b'] = MockInstanceConf(13.3, 4.3, 12.3, 3.0, 9.1)
    cnf.instances['c'] = MockInstanceConf(28.0, 13.2, 5.6, 7.4, 8.12)

    return cnf


class TestAggregatedConfusion:

    def test_correct(self, sample_confusion):
        return sample_confusion.correct == pytest.approx(10.2 + 13.3 + 28.0)

    def test_insertions(self, sample_confusion):
        return sample_confusion.insertions == pytest.approx(3.4 + 4.3 + 13.2)

    def test_deletions(self, sample_confusion):
        return sample_confusion.deletions == pytest.approx(2.9 + 12.3 + 5.6)

    def test_substitutions(self, sample_confusion):
        return sample_confusion.substitutions == pytest.approx(4.3 + 3.0 + 7.4)

    def test_substitutions_out(self, sample_confusion):
        return sample_confusion.substitutions_out == pytest.approx(1.2 + 9.1 + 8.12)
