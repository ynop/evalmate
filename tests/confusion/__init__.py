from audiomate import annotations

from evalmate import alignment
from evalmate import confusion

import pytest


def test_create_from_segments():
    segments = [
        alignment.Segment(0, 4, annotations.Label('music', start=0, end=5), annotations.Label('music', start=0, end=4)),
        alignment.Segment(4, 5, annotations.Label('music', start=0, end=5),
                          annotations.Label('speech', start=4, end=6)),
        alignment.Segment(5, 6, annotations.Label('speech', start=5, end=11),
                          annotations.Label('speech', start=4, end=6)),
        alignment.Segment(6, 8, annotations.Label('speech', start=5, end=11), None),
        alignment.Segment(8, 11, annotations.Label('speech', start=5, end=11),
                          annotations.Label('mix', start=8, end=16)),
        alignment.Segment(11, 14, annotations.Label('mix', start=11, end=14),
                          annotations.Label('mix', start=8, end=16)),
        alignment.Segment(14, 16, annotations.Label('speech', start=14, end=19),
                          annotations.Label('mix', start=8, end=16)),
        alignment.Segment(16, 19, annotations.Label('speech', start=14, end=19),
                          annotations.Label('speech', start=16, end=21)),
        alignment.Segment(19, 21, None, annotations.Label('speech', start=16, end=21))
    ]

    cnf = confusion.create_from_segments(segments)

    assert cnf.correct == pytest.approx(11)
    assert cnf.insertions == pytest.approx(2)
    assert cnf.deletions == pytest.approx(2)
    assert cnf.substitutions == pytest.approx(6)
    assert cnf.substitutions_out == pytest.approx(6)
    assert cnf.total == pytest.approx(19)

    assert len(cnf.instances) == 3

    assert cnf.instances['music'].correct == pytest.approx(4)
    assert cnf.instances['music'].insertions == pytest.approx(0)
    assert cnf.instances['music'].deletions == pytest.approx(0)
    assert cnf.instances['music'].substitutions == pytest.approx(1)
    assert cnf.instances['music'].substitutions_out == pytest.approx(0)
    assert cnf.instances['music'].total == pytest.approx(5)

    assert cnf.instances['speech'].correct == pytest.approx(4)
    assert cnf.instances['speech'].insertions == pytest.approx(2)
    assert cnf.instances['speech'].deletions == pytest.approx(2)
    assert cnf.instances['speech'].substitutions == pytest.approx(5)
    assert cnf.instances['speech'].substitutions_out == pytest.approx(1)
    assert cnf.instances['speech'].total == pytest.approx(11)

    assert cnf.instances['mix'].correct == pytest.approx(3)
    assert cnf.instances['mix'].insertions == pytest.approx(0)
    assert cnf.instances['mix'].deletions == pytest.approx(0)
    assert cnf.instances['mix'].substitutions == pytest.approx(0)
    assert cnf.instances['mix'].substitutions_out == pytest.approx(5)
    assert cnf.instances['mix'].total == pytest.approx(3)


def test_create_from_label_pairs():
    pairs = [
        alignment.LabelPair(annotations.Label('up', start=5.28, end=5.99),
                            annotations.Label('up', start=5.20, end=5.88)),
        alignment.LabelPair(annotations.Label('down', start=10.35, end=11.12),
                            annotations.Label('right', start=10.30, end=11.08)),
        alignment.LabelPair(annotations.Label('right', start=20.87, end=22.01), None),
        alignment.LabelPair(annotations.Label('up', start=33.00, end=33.4), None),
        alignment.LabelPair(annotations.Label('up', start=33.4, end=33.8), None),
        alignment.LabelPair(annotations.Label('down', start=39.28, end=40.0),
                            annotations.Label('down', start=39.27, end=40.01)),
        alignment.LabelPair(None, annotations.Label('up', start=32.00, end=32.5)),
        alignment.LabelPair(None, annotations.Label('up', start=34.2, end=34.8)),
        alignment.LabelPair(None, annotations.Label('left', start=39.3, end=39.9))
    ]

    cnf = confusion.create_from_label_pairs(pairs)

    assert 'up' in cnf.instances.keys()
    assert 'down' in cnf.instances.keys()
    assert 'right' in cnf.instances.keys()
    assert 'left' in cnf.instances.keys()

    assert cnf.instances['up'].correct == 1
    assert cnf.instances['up'].deletions == 2
    assert cnf.instances['up'].insertions == 2
    assert cnf.instances['up'].substitutions == 0
    assert cnf.instances['up'].substitutions_out == 0

    assert cnf.instances['down'].correct == 1
    assert cnf.instances['down'].deletions == 0
    assert cnf.instances['down'].insertions == 0
    assert cnf.instances['down'].substitutions == 1
    assert cnf.instances['down'].substitutions_out == 0

    assert cnf.instances['right'].correct == 0
    assert cnf.instances['right'].deletions == 1
    assert cnf.instances['right'].insertions == 0
    assert cnf.instances['right'].substitutions == 0
    assert cnf.instances['right'].substitutions_out == 1

    assert cnf.instances['left'].correct == 0
    assert cnf.instances['left'].deletions == 0
    assert cnf.instances['left'].insertions == 1
    assert cnf.instances['left'].substitutions == 0
    assert cnf.instances['left'].substitutions_out == 0
