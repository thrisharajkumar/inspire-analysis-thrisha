# Pins the SMOTENC bug: imbalanced-learn 0.14.x rejects a boolean categorical mask with
# "truth value of an array ... is ambiguous", which the original notebook's bare
# `except ValueError` swallowed for EVERY stratum. Integer indices work.
import numpy as np
import pytest
from imblearn.over_sampling import SMOTENC


def _data():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=(60, 3)), rng.integers(0, 2, (60, 2))]
    y = np.r_[np.ones(10), np.zeros(50)]
    return X, y, np.array([False, False, False, True, True])


def test_integer_indices_generate_synthetic_rows():
    X, y, mask = _data()
    idx = np.where(mask)[0].tolist()
    Xr, yr = SMOTENC(categorical_features=idx, sampling_strategy={1: 20}, k_neighbors=3,
                     random_state=0).fit_resample(X, y)
    assert len(Xr) - len(X) == 10 and np.all(yr[len(X):] == 1)
    # categorical columns of synthetic rows are real category values, not interpolated
    assert set(np.unique(Xr[len(X):, 3:])) <= {0.0, 1.0}


def test_boolean_mask_behaviour_is_known():
    # Documents the library behaviour the fix works around. If a future imblearn accepts
    # the mask again this test will say so -- the fix (indices) stays correct either way.
    X, y, mask = _data()
    try:
        SMOTENC(categorical_features=mask, sampling_strategy={1: 20}, k_neighbors=3,
                random_state=0).fit_resample(X, y)
    except ValueError as e:
        assert "ambiguous" in str(e)
    else:
        pytest.skip("this imbalanced-learn version accepts boolean masks")
