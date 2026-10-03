# SHAP-style attribution INSIDE each risk term of the INSPIRE organ-system DNN (v4).
#
# The model already says HOW MUCH each organ system added to a patient's risk (the NAM
# terms). This module answers the next question: WHICH of that system's own measurements
# made it say so ("the kidney points came mostly from creatinine, a little from potassium").
#
# Method: Expected Gradients (Erion et al., 2021) -- the estimator behind shap's
# GradientExplainer. For one risk term f (e.g. the kidney contribution) and one patient x:
#
#     attribution_j  =  E_{baseline b ~ background, alpha ~ U(0,1)} [ (x_j - b_j) * d f(b + alpha (x - b)) / d x_j ]
#
# Only the inputs we are explaining are moved from the baseline patient towards x; every
# other input stays at x's own value. The attributions then add up (approximately, it is a
# Monte-Carlo estimate) to  f(x) - E_b[f(x with those inputs taken from b)]  -- "how far this
# patient's term is from a typical patient's, split across the inputs". `completeness_gap`
# reports how close the sum is, as a sanity check.
#
# Pure PyTorch, no dataset globals, no extra dependency (the `shap` package is not needed).
import torch


def _gather(t, idx):
    return t[idx] if t is not None else None


def expected_gradients(model, batch, background, term, ts_systems=(), static_cols=(),
                       n_samples=32, seed=0, check_completeness=True):
    """Attribute ONE risk term to a chosen set of inputs, per patient.

    model        -- a MortalityModel with NAM fusion (per-term contributions must exist)
    batch        -- dict of tensors for the B patients to explain (already on the model's device)
    background   -- dict of tensors for Nb reference patients (same keys), the "typical patient" pool
    term         -- one of model.term_names, e.g. "renal", "whole_patient", "surgical_context"
    ts_systems   -- organ systems whose time series (values AND missing-pattern mask) are attributed
    static_cols  -- static-vector column indices that are attributed
    Returns {"ts": {system: [B, F] attributions summed over the 24 time points},
             "static": [B, len(static_cols)], "f_x": [B], "f_base_mean": [B], "completeness_gap": float}
    """
    model.eval()
    dev = batch["static"].device
    B = batch["static"].shape[0]
    Nb = background["static"].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    static_cols = list(static_cols)
    col_mask = torch.zeros(batch["static"].shape[1], dtype=torch.bool, device=dev)
    if static_cols:
        col_mask[static_cols] = True

    acc_ts = {s: torch.zeros(B, batch[f"{s}_x"].shape[-1], device=dev) for s in ts_systems}
    acc_st = torch.zeros(B, len(static_cols), device=dev)
    f_base_sum = torch.zeros(B, device=dev)

    def term_value(out):
        sig = out["per_system_signal"]
        if sig is None or term not in sig:
            raise ValueError(f"term {term!r} not available (needs NAM fusion; terms: {list(sig or {})})")
        return sig[term]

    for _ in range(n_samples):
        idx = torch.randint(0, Nb, (B,), generator=gen).to(dev)
        alpha = torch.rand(B, generator=gen).to(dev)
        b = dict(batch)
        leaves, diffs = [], []
        for s in ts_systems:
            for kind in ("x", "mask"):
                x = batch[f"{s}_{kind}"]
                base = background[f"{s}_{kind}"][idx]
                a = alpha.view(B, 1, 1)
                leaf = (base + a * (x - base)).detach().requires_grad_(True)
                b[f"{s}_{kind}"] = leaf
                leaves.append(leaf); diffs.append((s, kind, x - base))
        if static_cols:
            x = batch["static"]
            base = background["static"][idx]
            mixed = torch.where(col_mask.view(1, -1), base + alpha.view(B, 1) * (x - base), x)
            st_leaf = mixed.detach().requires_grad_(True)
            b["static"] = st_leaf
            leaves.append(st_leaf); diffs.append(("static", None, x - base))
        f = term_value(model(b))
        grads = torch.autograd.grad(f.sum(), leaves, allow_unused=True)
        for g, (s, kind, d) in zip(grads, diffs):
            if g is None:
                continue
            contrib = g * d
            if s == "static":
                acc_st += contrib[:, static_cols]
            else:
                acc_ts[s] += contrib.sum(dim=1)          # sum over time points -> [B, F]
        if check_completeness:
            with torch.no_grad():
                b0 = dict(batch)
                for s in ts_systems:
                    for kind in ("x", "mask"):
                        b0[f"{s}_{kind}"] = background[f"{s}_{kind}"][idx]
                if static_cols:
                    b0["static"] = torch.where(col_mask.view(1, -1), background["static"][idx], batch["static"])
                f_base_sum += term_value(model(b0))

    res = {"ts": {s: (acc_ts[s] / n_samples).detach() for s in ts_systems},
           "static": (acc_st / n_samples).detach()}
    with torch.no_grad():
        f_x = term_value(model(batch))
    res["f_x"] = f_x.detach()
    if check_completeness:
        f_base = f_base_sum / n_samples
        total = sum(v.sum(dim=1) for v in res["ts"].values()) + res["static"].sum(dim=1) if (res["ts"] or static_cols) \
            else torch.zeros(B, device=dev)
        target = f_x - f_base
        res["f_base_mean"] = f_base.detach()
        denom = target.abs().mean().clamp_min(1e-8)
        res["completeness_gap"] = float((total - target).abs().mean() / denom)
    return res
