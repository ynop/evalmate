from evalmate import confusion

import pytest


@pytest.fixture
def sample_confusion():
    class ConfMock(confusion.Confusion):

        @property
        def correct(self):
            return 19.9

        @property
        def insertions(self):
            return 28.2

        @property
        def deletions(self):
            return 12.39

        @property
        def substitutions(self):
            return 13.1

        @property
        def substitutions_out(self):
            return 21.9

    return ConfMock()


class TestConfusion:

    def test_total(self, sample_confusion):
        assert sample_confusion.total == pytest.approx(19.9 + 12.39 + 13.1)

    def test_false_negatives(self, sample_confusion):
        assert sample_confusion.false_negatives == pytest.approx(12.39 + 13.1)

    def test_false_positives(self, sample_confusion):
        assert sample_confusion.false_positives == pytest.approx(28.2 + 21.9)

    def test_true_positives(self, sample_confusion):
        assert sample_confusion.true_positives == pytest.approx(19.9)

    def test_error_rate(self, sample_confusion):
        assert sample_confusion.error_rate == pytest.approx((28.2 + 12.39 + 13.1) / (19.9 + 12.39 + 13.1))

    def test_accuracy(self, sample_confusion):
        assert sample_confusion.accuracy == pytest.approx(19.9 / (19.9 + 12.39 + 13.1 + 28.2))

    def test_precision(self, sample_confusion):
        assert sample_confusion.precision == pytest.approx(19.9 / (19.9 + 28.2 + 21.9))

    def test_recall(self, sample_confusion):
        assert sample_confusion.recall == pytest.approx(19.9 / (19.9 + 12.39 + 13.1))

    def test_f_measure(self, sample_confusion):
        assert sample_confusion.f_measure() == pytest.approx(0.3449172372)

    def test_f_2_measure(self, sample_confusion):
        assert sample_confusion.f_measure(beta=2) == pytest.approx(0.395531881)
