# Wiring checks for the v3 model -- random tensors, no data needed.
import torch
from inspire_dnn.layers import (MortalityModel, SystemCouplingLayer, SystemArbitrationLayer, WholePatientLayer,
                                ENCODER_SIZE_PRESETS, coupling_effect, own_data_share)

TS = {"renal": 10, "cardiovascular": 8, "respiratory": 11, "metabolic_hepatic": 10,
      "haematology": 10, "neurological": 3, "gi": 0, "msk": 0}
N_STATIC = 40
SYS_IDX = {s: [2 * i, 2 * i + 1] for i, s in enumerate(TS)}          # 2 own static columns per system
CONTEXT = list(range(16, N_STATIC))


def _batch(B=6, T=24, seed=0):
    torch.manual_seed(seed)
    b = {"static": torch.randn(B, N_STATIC)}
    for s, f in TS.items():
        if f:
            b[f"{s}_x"] = torch.randn(B, T, f)
            b[f"{s}_mask"] = (torch.rand(B, T, f) > 0.5).float()
    return b


def _model(**kw):
    d, h, l, ff = ENCODER_SIZE_PRESETS["medium"]
    args = dict(embed_dim=d, n_heads=h, n_layers=l, ff_multiplier=ff)
    args.update(kw)
    torch.manual_seed(1)
    return MortalityModel(TS, SYS_IDX, CONTEXT, N_STATIC, **args).eval()


def test_all_eight_systems_are_organ_systems_with_own_terms():
    m = _model()
    out = m(_batch())
    assert m.organ_systems == list(TS)
    assert set(out["per_system_signal"]) == set(TS) | {"whole_patient"}
    assert set(out["system_say"]) == set(TS)                           # GI and MSK get a say too
    assert out["coupling_report"]["weights"].shape == (6, 8, 8)


def test_exact_additive_decomposition():
    m = _model()
    out = m(_batch())
    assert torch.allclose(sum(out["per_system_signal"].values()) + m.fusion.bias, out["logit"], atol=1e-5)
    for s, w in out["system_say"].items():
        assert torch.allclose(out["per_system_signal"][s], w * out["system_answers"][s], atol=1e-6)


def test_shared_body_has_far_fewer_parameters():
    shared = sum(p.numel() for p in _model(share_encoder=True).encoder.parameters())
    separate = sum(p.numel() for p in _model(share_encoder=False).encoder.parameters())
    assert shared * 3 < separate


def test_whole_patient_layer_starts_as_no_adjustment_and_is_bounded():
    wp = WholePatientLayer(N_STATIC, TS, list(TS), 32)
    g, scales = wp(_batch())
    assert torch.allclose(scales, torch.ones_like(scales))
    with torch.no_grad():
        wp.film.weight.normal_(0, 10.0)
    _, scales = wp(_batch())
    assert scales.min() >= 0.5 - 1e-6 and scales.max() <= 1.5 + 1e-6


def test_system_answer_depends_on_own_data_at_init():
    # at initialisation (scales = 1, coupling gate small) each system's answer should be
    # driven mostly by its own inputs -- the leakage check must be able to see that.
    res = own_data_share(_model(), _batch(B=32))
    for s in ("renal", "cardiovascular", "gi"):
        assert res[s]["own_share"] > 0.5, (s, res[s])


def test_masked_system_pretraining_trains_the_coupling_layer():
    m = _model().train()
    loss, parts = m.pretrain_loss(_batch(B=16))
    loss.backward()
    assert parts["masked_system"] > 0
    assert m.coupling.q.weight.grad is not None and m.coupling.q.weight.grad.abs().sum() > 0
    assert m.mask_token.grad is not None


def test_coupling_identity_option_and_no_self_attention():
    layer = SystemCouplingLayer(["a", "b", "c"], 16, n_heads=2, gate_init=0.0).eval()
    E = torch.randn(4, 3, 16)
    out, rep = layer(E)
    assert torch.allclose(out, E)
    assert torch.allclose(rep["weights"].diagonal(dim1=-2, dim2=-1), torch.zeros(4, 3))


def test_arbitration_starts_equal_say():
    w, aux = SystemArbitrationLayer(["a", "b", "c", "d"], 8)(torch.randn(6, 8))
    assert torch.allclose(w.detach(), torch.ones(6, 4)) and float(aux) == 0.0


def test_ablation_switches_run():
    for kw in (dict(coupling_mode="none"), dict(use_arbitration=False), dict(use_whole_patient=False),
               dict(fusion_strategy="concat"), dict(share_encoder=False)):
        out = _model(**kw)(_batch())
        assert out["logit"].shape == (6,)
    assert coupling_effect(_model(coupling_mode="none"), _model(coupling_mode="none")(_batch())) is None


def test_system_dropout_only_in_training():
    m = _model(system_dropout=0.5)
    b = _batch()
    assert torch.allclose(m(b)["logit"], m(b)["logit"])            # eval: deterministic
    m.train(); torch.manual_seed(3)
    l1 = m(b)["logit"]; torch.manual_seed(4); l2 = m(b)["logit"]
    assert not torch.allclose(l1, l2)


def test_end_to_end_training_step():
    m = _model(system_dropout=0.1).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=1e-2)
    b, y = _batch(B=16), (torch.rand(16) > 0.5).float()
    for _ in range(3):
        opt.zero_grad()
        out = m(b)
        (torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], y) + 1e-3 * out["aux_loss"]).backward()
        opt.step()
    assert m.arbitration.net[-1].weight.abs().sum() > 0
    assert m.whole_patient.film.weight.abs().sum() > 0


# ---------------------------------------------------------------- v4: surgical-context term
SC = [16, 17, 18, 19]   # e.g. emop, asa, dept_*, antype_* columns


def test_surgical_context_is_its_own_term_and_still_exact():
    m = _model(surgical_context_idx=SC)
    out = m(_batch())
    assert m.term_names[-1] == "surgical_context"
    assert set(out["per_system_signal"]) == set(TS) | {"whole_patient", "surgical_context"}
    recon = sum(out["per_system_signal"].values()) + m.fusion.bias
    assert torch.allclose(recon, out["logit"], atol=1e-5)          # still an exact breakdown
    assert out["surgical_context_embedding"].shape == (6, ENCODER_SIZE_PRESETS["medium"][0])


def test_surgical_context_exclusive_hides_columns_from_whole_patient():
    m = _model(surgical_context_idx=SC, surgical_context_exclusive=True)
    b = _batch(); b2 = {k: v.clone() for k, v in b.items()}
    b2["static"][:, SC] += 5.0                                      # change ONLY the surgical columns
    with torch.no_grad():
        g1 = m(b)["static_embedding"]; g2 = m(b2)["static_embedding"]
    assert torch.allclose(g1, g2)                                  # whole-patient summary did not move
    m2 = _model(surgical_context_idx=SC, surgical_context_exclusive=False)
    with torch.no_grad():
        assert not torch.allclose(m2(b)["static_embedding"], m2(b2)["static_embedding"])


def test_surgical_context_works_with_coupling_effect_and_without_whole_patient():
    m = _model(surgical_context_idx=SC)
    out = m(_batch())
    eff = coupling_effect(m, out)
    assert set(eff) == set(TS)
    m2 = _model(surgical_context_idx=SC, use_whole_patient=False)
    assert m2(_batch())["logit"].shape == (6,)


def test_dropping_gi_and_msk_builds_a_six_system_model():
    keep = {s: f for s, f in TS.items() if s not in ("gi", "msk")}
    d, h, l, ff = ENCODER_SIZE_PRESETS["small"]
    m = MortalityModel(keep, {s: SYS_IDX[s] for s in keep}, CONTEXT + SYS_IDX["gi"] + SYS_IDX["msk"], N_STATIC,
                       embed_dim=d, n_heads=h, n_layers=l, ff_multiplier=ff).eval()
    out = m(_batch())
    assert m.organ_systems == list(keep) and out["coupling_report"]["weights"].shape == (6, 6, 6)


def test_big_presets_build():
    for size in ("xlarge", "xxlarge"):
        d, h, l, ff = ENCODER_SIZE_PRESETS[size]
        m = MortalityModel(TS, SYS_IDX, CONTEXT, N_STATIC, embed_dim=d, n_heads=h, n_layers=l, ff_multiplier=ff).eval()
        assert m(_batch(B=2))["logit"].shape == (2,)
