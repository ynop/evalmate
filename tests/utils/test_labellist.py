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
