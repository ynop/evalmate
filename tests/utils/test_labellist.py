from audiomate import annotations

from evalmate.utils import labellist


def test_close_pairs(kws_ref_and_hyp_label_list):
    ll_ref, ll_hyp = kws_ref_and_hyp_label_list

    matches, no_ref_match, no_hyp_match = labellist.close_pairs(ll_ref, ll_hyp,
                                                                start_delta_threshold=0.2,
                                                                end_delta_threshold=-1)
    expected_matches = [
        (0, 0),
        (1, 1),
        (5, 4),
        (5, 5)
    ]

    assert sorted(expected_matches) == sorted(matches)
    assert {2, 3, 4} == no_ref_match
    assert {2, 3} == no_hyp_match


def test_overlapping_pairs():
    ll_ref = annotations.LabelList(labels=[
        annotations.Label('a', 2.2, 3.4),
        annotations.Label('c', 19.3, 33.0),
        annotations.Label('b', 5.0, 8.43)
    ])

    ll_hyp = annotations.LabelList(labels=[
        annotations.Label('x', 2.0, 3.0),
        annotations.Label('y', 3.3, 4.5),
        annotations.Label('z', 6.3, 8.2),
        annotations.Label('w', 39.0, 44.3)
    ])

    pairs, ref_rest, hyp_rest = labellist.overlapping_pairs(ll_ref, ll_hyp, min_overlap=0.1)

    assert sorted(pairs) == sorted([
        (0, 0),
        (0, 1),
        (2, 2)
    ])

    assert ref_rest == {1}
    assert hyp_rest == {3}
