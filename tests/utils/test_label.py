from audiomate.corpus import assets

import pytest

from evalmate.utils import label


def test_overlap_percentage():
    overlap = label.overlap_percentage(assets.Label('a', start=1.45, end=5.33),
                                       assets.Label('b', start=2.49, end=4.92))
    assert overlap == pytest.approx(2.43 / (5.33 - 1.45))


def test_overlap_percentage_of_non_overlapping():
    overlap = label.overlap_percentage(assets.Label('a', start=1.45, end=5.33),
                                       assets.Label('b', start=7.3, end=8.7))
    assert overlap == 0
