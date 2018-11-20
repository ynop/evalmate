from audiomate import annotations

import pytest

from evalmate.utils import label


def test_overlap_percentage():
    overlap = label.overlap_percentage(annotations.Label('a', start=1.45, end=5.33),
                                       annotations.Label('b', start=2.49, end=4.92))
    assert overlap == pytest.approx(2.43 / (5.33 - 1.45))


def test_overlap_percentage_of_non_overlapping():
    overlap = label.overlap_percentage(annotations.Label('a', start=1.45, end=5.33),
                                       annotations.Label('b', start=7.3, end=8.7))
    assert overlap == 0


def test_overlap_time_with_no_overlap():
    overlap = label.overlap_time(annotations.Label('a', start=1.45, end=2.23),
                                 annotations.Label('b', start=2.49, end=4.92))
    assert overlap == 0.0


def test_overlap_time_with_full_overlap_ref():
    overlap = label.overlap_time(annotations.Label('a', start=2.5, end=4.3),
                                 annotations.Label('b', start=2.49, end=4.92))
    assert overlap == pytest.approx(4.3 - 2.5)


def test_overlap_time_with_full_overlap_hyp():
    overlap = label.overlap_time(annotations.Label('a', start=1.45, end=5.33),
                                 annotations.Label('b', start=2.49, end=4.92))
    assert overlap == pytest.approx(4.92 - 2.49)


def test_overlap_time_with_right_overlap():
    overlap = label.overlap_time(annotations.Label('a', start=1.45, end=2.23),
                                 annotations.Label('b', start=2.11, end=4.92))
    assert overlap == pytest.approx(2.23 - 2.11)


def test_overlap_time_with_left_overlap():
    overlap = label.overlap_time(annotations.Label('a', start=1.45, end=2.23),
                                 annotations.Label('b', start=0.04, end=2.01))
    assert overlap == pytest.approx(2.01 - 1.45)


def test_overlap_time_with_endless_label():
    overlap = label.overlap_time(annotations.Label('a', start=1.45, end=2.23),
                                 annotations.Label('b', start=0.04, end=-1))
    assert overlap == pytest.approx(2.23 - 1.45)
