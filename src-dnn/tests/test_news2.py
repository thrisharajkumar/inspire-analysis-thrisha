# NEWS2 band edges against the RCP 2017 chart, plus the pipeline-specific mappings.
import numpy as np
import pytest
from inspire_dnn import news2 as n2


def test_self_test_passes():
    ok, failures = n2.news2_self_test()
    assert ok, failures


@pytest.mark.parametrize("fn,x,want", [
    (n2.score_respiration_rate, 8, 3), (n2.score_respiration_rate, 9, 1), (n2.score_respiration_rate, 12, 0),
    (n2.score_respiration_rate, 20, 0), (n2.score_respiration_rate, 21, 2), (n2.score_respiration_rate, 25, 3),
    (n2.score_spo2_scale1, 91, 3), (n2.score_spo2_scale1, 92, 2), (n2.score_spo2_scale1, 94, 1), (n2.score_spo2_scale1, 96, 0),
    (n2.score_systolic_bp, 90, 3), (n2.score_systolic_bp, 91, 2), (n2.score_systolic_bp, 101, 1),
    (n2.score_systolic_bp, 111, 0), (n2.score_systolic_bp, 219, 0), (n2.score_systolic_bp, 220, 3),
    (n2.score_pulse, 40, 3), (n2.score_pulse, 41, 1), (n2.score_pulse, 51, 0), (n2.score_pulse, 91, 1),
    (n2.score_pulse, 111, 2), (n2.score_pulse, 130, 2), (n2.score_pulse, 131, 3),
    (n2.score_temperature, 35.0, 3), (n2.score_temperature, 35.1, 1), (n2.score_temperature, 36.1, 0),
    (n2.score_temperature, 38.1, 1), (n2.score_temperature, 39.0, 1), (n2.score_temperature, 39.1, 2),
])
def test_band_edges(fn, x, want):
    assert float(fn(x)) == want


def test_vectorised_and_nan_safe():
    out = n2.score_pulse([np.nan, 72, 135])
    assert np.isnan(out[0]) and out[1] == 0 and out[2] == 3


def test_max_possible_total_is_20():
    c = n2.news2_components(rr=5, spo2=80, on_oxygen=1, sbp=80, hr=150, alert=0, temp_c=34)
    total, n_core, any3 = n2.news2_total(c)
    assert float(total) == 20 and int(n_core) == 5


def test_observation_sets_bounded_carry_forward():
    series = {"hr": {0: 72, 60: 120, 600: 135}, "rr": {0: 16, 60: 22}, "nibp_sbp": {0: 120, 60: 95},
              "bt": {0: 36.8}, "spo2": {60: 93}}
    sets = n2.news2_observation_sets(series, bucket_minutes=60, carry_forward_minutes=240)
    assert list(sets["total"][:2]) == [0.0, 8.0]
    assert np.isnan(sets["total"][2])   # 9 h later only HR is fresh -> no score, not a falsely low one


def test_summary_features_missing_patient():
    feats = n2.news2_summary_features(n2.news2_observation_sets({}))
    assert feats["news2_available"] == 0.0 and np.isnan(feats["news2_max"])


def test_compare_before_after_flags_out_of_range():
    res = n2.compare_before_after([2, 3, 4, 5], [3, 4, 25])
    assert res["verdict"].startswith("FAIL")
    res = n2.compare_before_after(np.arange(0, 10, 0.5), np.arange(0.2, 10, 0.5))
    assert res["verdict"] == "PASS"
