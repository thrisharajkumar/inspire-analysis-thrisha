# v4: SHAP-style expected gradients inside one risk term -- random tensors, no data needed.
import torch
from inspire_dnn.layers import MortalityModel, ENCODER_SIZE_PRESETS
from inspire_dnn.explain import expected_gradients
from test_layers import TS, SYS_IDX, CONTEXT, N_STATIC, _batch

def _m():
    d, h, l, ff = ENCODER_SIZE_PRESETS["small"]
    torch.manual_seed(3)
    return MortalityModel(TS, SYS_IDX, CONTEXT, N_STATIC, embed_dim=d, n_heads=h, n_layers=l, ff_multiplier=ff,
                          surgical_context_idx=[16, 17, 18]).eval()


def test_shapes_and_completeness():
    m = _m(); b = _batch(B=5, seed=1); bg = _batch(B=20, seed=2)
    r = expected_gradients(m, b, bg, "renal", ts_systems=["renal"], static_cols=SYS_IDX["renal"], n_samples=64)
    assert r["ts"]["renal"].shape == (5, TS["renal"]) and r["static"].shape == (5, 2)
    assert r["completeness_gap"] < 0.35          # Monte-Carlo estimate: roughly adds up


def test_static_only_terms():
    m = _m(); b = _batch(B=4, seed=1); bg = _batch(B=10, seed=2)
    r = expected_gradients(m, b, bg, "surgical_context", static_cols=[16, 17, 18], n_samples=16)
    assert r["static"].shape == (4, 3) and r["ts"] == {}
    r2 = expected_gradients(m, b, bg, "gi", static_cols=SYS_IDX["gi"], n_samples=8)   # a system with no time series
    assert r2["static"].shape == (4, 2)


def test_inputs_with_no_effect_get_zero():
    m = _m(); b = _batch(B=4, seed=1); bg = _batch(B=10, seed=2)
    # the surgical-context term cannot see renal columns -> zero attribution
    r = expected_gradients(m, b, bg, "surgical_context", static_cols=SYS_IDX["renal"], n_samples=8)
    assert torch.allclose(r["static"], torch.zeros_like(r["static"]), atol=1e-7)
