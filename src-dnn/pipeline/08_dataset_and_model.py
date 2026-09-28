# %% [markdown]
# # Part 9 — The multimodal organ-system DNN
#
# This is where every theory decision from Part 1 becomes a `torch.nn.Module`. Build order:
#
# - §9.1 the `INSPIREDataset` — turns `IMPUTED_BUNDLE` into model-ready tensors
# - §9.2 `SystemEncoder` — one per organ system, transformer over time (§1.5.1) with the
#   §6.9 cardiovascular→renal coupling wired in via an extra side-input
# - §9.3 `StaticEncoder` — the small MLP for static/GI/MSK/HFRS/operation-count features
#   (§1.5.1, §1.6.1)
# - §9.4 the autoencoder pre-training head (§1.5.4, phase 1 of "jointly learned")
# - §9.5 the loss functions — `pos_weight`-scaled BCE and focal loss (§1.4.1)
# - §9.6 `MortalityModel` — the full network: six system encoders + static encoder → fusion
# - §9.7 the fusion head itself — NAM-style additive (default) or plain concatenation
#   (§1.5.2), switchable via `CONFIG['FUSION_STRATEGY']`

# %% [markdown]
# ## 9.1 `INSPIREDataset` — bundle → tensors
#
# One `__getitem__` call returns everything the model needs for one patient: six
# `[T, F_system]` tensors + six masks, the static feature vector, and the label. Three kinds
# of item now feed the training dataset, each flagged in the returned dict so later analysis
# (§11's per-patient qualitative checks) can tell them apart:
#
# - **Real patients** — straight from `IMPUTED_BUNDLE`.
# - **Synthetic static rows** (§8.2, grouped SMOTENC + Tomek) — a real static vector, but no
#   real time series exists for them: every system's sequence is filled with zeros (the
#   population mean, on the standardized scale — §7.6) and the mask is all-zero, honestly
#   recording that nothing was actually observed.
# - **Augmented real minority patients** (§8.3) — a *real* patient's *real* static vector
#   and diagnosis history, but with jitter and/or a time-mask applied to their sequence
#   branch on the fly, a different random perturbation each epoch (`copy_idx` combined with
#   the current epoch seeds the perturbation, so repeated passes over the same augmented
#   item don't produce an identical duplicate every time).

# %%
import zlib

def _stable_seed(sid, copy_idx, epoch_seed):
    # v2: Python's built-in hash() of a string is randomised per process (PYTHONHASHSEED),
    # so v1's augmentation noise was not reproducible between sessions. crc32 is.
    return zlib.crc32(f"{sid}|{copy_idx}|{epoch_seed}".encode()) & 0xFFFFFFFF

def _jitter_cols(system):
    # 1.0 for features that may be jittered, 0.0 for discrete/ordinal/binary ones (v2)
    fnames = PATIENT_BUNDLE[TRAIN_IDS[0]]["ts"][system]["feature_names"]
    return np.array([0.0 if f in CONFIG["JITTER_EXCLUDE_ITEMS"] else 1.0 for f in fnames])

JITTER_COLS = {s: _jitter_cols(s) for s in TIME_SERIES_SYSTEMS}

def _jitter_and_mask_sequence(raw, mask, sigma, rng, jitter_cols=None):
    # raw, mask: [T, F] numpy arrays, already imputed + standardized (Part 7).
    # Jitter only OBSERVED positions (mask==1) so noise doesn't compound with imputation.
    # Time-mask: zero out a short contiguous span of timesteps (all features at once) --
    # the fixed-length-grid stand-in for window slicing/cropping, see Part 8.3.
    if raw.shape[1] == 0:
        return raw.copy(), mask.copy()
    out_raw = raw.copy()
    out_mask = mask.copy()

    noise = rng.normal(loc=0.0, scale=sigma, size=out_raw.shape)
    if jitter_cols is not None:
        noise = noise * jitter_cols[None, :]   # v2: leave discrete items (GCS, device flags) untouched
    out_raw = out_raw + noise * out_mask   # only perturb where mask==1 (real observations)

    T = out_raw.shape[0]
    if T >= 4:
        span = max(1, T // 6)
        start = rng.integers(0, max(T - span, 1))
        out_raw[start:start + span, :] = 0.0
        out_mask[start:start + span, :] = 0.0

    return out_raw, out_mask

class INSPIREDataset(Dataset):
    def __init__(self, subject_ids, synthetic_static_rows=None, augment_items=None, epoch_seed=0,
                 synthetic_ts_donors=None):
        self.subject_ids = list(subject_ids)
        self.synthetic_rows = synthetic_static_rows or []
        self.synthetic_ts_donors = synthetic_ts_donors or []   # v2: aligned with synthetic_rows (Part 8.2c)
        self.augment_items = augment_items or []   # list of (sid, copy_idx)
        self.n_real = len(self.subject_ids)
        self.n_synthetic = len(self.synthetic_rows)
        self.epoch_seed = epoch_seed   # bump this between epochs for fresh augmentation noise

    def __len__(self):
        return self.n_real + self.n_synthetic + len(self.augment_items)

    def __getitem__(self, idx):
        if idx < self.n_real:
            sid = self.subject_ids[idx]
            item = {"is_synthetic": False, "is_augmented": False, "subject_id": sid}
            for system in TIME_SERIES_SYSTEMS:
                raw = IMPUTED_BUNDLE[sid][system]
                mask = PATIENT_BUNDLE[sid]["ts"][system]["mask"]
                item[f"{system}_x"] = torch.tensor(raw, dtype=torch.float32)
                item[f"{system}_mask"] = torch.tensor(mask, dtype=torch.float32)
            item["static"] = torch.tensor(IMPUTED_BUNDLE[sid]["static"].values.astype(float), dtype=torch.float32)
            item["label"] = torch.tensor(PATIENT_BUNDLE[sid]["label"], dtype=torch.float32)
            return item

        idx -= self.n_real
        if idx < self.n_synthetic:
            static_vec, label = self.synthetic_rows[idx]
            item = {"is_synthetic": True, "is_augmented": False, "subject_id": f"synthetic_{idx}"}
            if self.synthetic_ts_donors:
                # v2: borrow the nearest real died patient's series, freshly jittered each epoch
                donor = self.synthetic_ts_donors[idx]
                rng = np.random.default_rng(_stable_seed(f"synthetic_{idx}", 0, self.epoch_seed))
                for system in TIME_SERIES_SYSTEMS:
                    aug_raw, aug_mask = _jitter_and_mask_sequence(
                        IMPUTED_BUNDLE[donor][system], PATIENT_BUNDLE[donor]["ts"][system]["mask"],
                        CONFIG["JITTER_SIGMA"], rng, JITTER_COLS[system])
                    item[f"{system}_x"] = torch.tensor(aug_raw, dtype=torch.float32)
                    item[f"{system}_mask"] = torch.tensor(aug_mask, dtype=torch.float32)
                item["static"] = torch.tensor(static_vec, dtype=torch.float32)
                item["label"] = torch.tensor(label, dtype=torch.float32)
                return item
            for system in TIME_SERIES_SYSTEMS:
                fnames = PATIENT_BUNDLE[self.subject_ids[0]]["ts"][system]["feature_names"]
                # NOTE: IMPUTED_BUNDLE is standardised (§7.6, z-score, mean 0) by the time this
                # runs, so "population mean" for a synthetic patient is simply 0 in this space
                # -- not the raw STATS mean, which is still in original clinical units.
                filled = np.zeros((CONFIG["TARGET_SEQ_LEN"], len(fnames))) if fnames else np.zeros((CONFIG["TARGET_SEQ_LEN"], 0))
                item[f"{system}_x"] = torch.tensor(filled, dtype=torch.float32)
                item[f"{system}_mask"] = torch.zeros(filled.shape, dtype=torch.float32)   # honestly: nothing was observed
            item["static"] = torch.tensor(static_vec, dtype=torch.float32)
            item["label"] = torch.tensor(label, dtype=torch.float32)
            return item

        idx -= self.n_synthetic
        sid, copy_idx = self.augment_items[idx]
        rng = np.random.default_rng(_stable_seed(sid, copy_idx, self.epoch_seed))
        item = {"is_synthetic": False, "is_augmented": True, "subject_id": f"{sid}_aug{copy_idx}"}
        for system in TIME_SERIES_SYSTEMS:
            raw = IMPUTED_BUNDLE[sid][system]
            mask = PATIENT_BUNDLE[sid]["ts"][system]["mask"]
            aug_raw, aug_mask = _jitter_and_mask_sequence(raw, mask, CONFIG["JITTER_SIGMA"], rng, JITTER_COLS[system])
            item[f"{system}_x"] = torch.tensor(aug_raw, dtype=torch.float32)
            item[f"{system}_mask"] = torch.tensor(aug_mask, dtype=torch.float32)
        item["static"] = torch.tensor(IMPUTED_BUNDLE[sid]["static"].values.astype(float), dtype=torch.float32)
        item["label"] = torch.tensor(PATIENT_BUNDLE[sid]["label"], dtype=torch.float32)
        return item

def collate_bundle(batch):
    out = {}
    for key in batch[0]:
        if key in ("is_synthetic", "is_augmented", "subject_id"):
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out

train_dataset = INSPIREDataset(EFFECTIVE_TRAIN_IDS, SYNTHETIC_STATIC_ROWS, AUGMENT_ITEMS,
                               synthetic_ts_donors=SYNTHETIC_TS_DONORS)
val_dataset   = INSPIREDataset(VAL_IDS)
test_dataset  = INSPIREDataset(TEST_IDS)

from torch.utils.data import WeightedRandomSampler

def item_labels(ds):
    # label of every dataset item: real patients, then synthetic (all 1), then augmented copies (all 1)
    return np.array([PATIENT_BUNDLE[s]["label"] for s in ds.subject_ids] + [1.0] * ds.n_synthetic
                    + [1.0] * len(ds.augment_items))

def make_train_loader(ds, seed=SEED):
    # v3: balanced batches. At ~1:200 a plain shuffled batch of 256 has ~1.25 deaths on
    # average and ~29% of batches have none. The sampler draws each epoch's items so that
    # ~SAMPLER_TARGET_POSITIVE_RATE of them are deaths (real, synthetic or augmented).
    bs = min(CONFIG["BATCH_SIZE"], len(ds))
    if not CONFIG["BALANCED_SAMPLER"]:
        return DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_bundle)
    y = item_labels(ds)
    p = CONFIG["SAMPLER_TARGET_POSITIVE_RATE"]
    n_pos, n_neg = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    w = np.where(y == 1, p / n_pos, (1 - p) / n_neg)
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(ds), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    return DataLoader(ds, batch_size=bs, sampler=sampler, collate_fn=collate_bundle)

train_loader = make_train_loader(train_dataset)
# Pre-training is label-free, so it uses REAL training patients only, in their natural mix
# (synthetic/augmented copies of deaths would only distort what "normal" looks like).
pretrain_dataset = INSPIREDataset(EFFECTIVE_TRAIN_IDS)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=min(CONFIG["BATCH_SIZE"], len(pretrain_dataset)),
                             shuffle=True, collate_fn=collate_bundle)
if CONFIG["BALANCED_SAMPLER"]:
    # Batches are now ~10% deaths, so the loss weight that finishes the balancing is the
    # batch-level ratio, not the raw training ratio (recalibration in 11.1c maps the scores
    # back to the true risk scale).
    POS_WEIGHT = (1 - CONFIG["SAMPLER_TARGET_POSITIVE_RATE"]) / CONFIG["SAMPLER_TARGET_POSITIVE_RATE"]
    print(f"Balanced sampler on: ~{CONFIG['SAMPLER_TARGET_POSITIVE_RATE']:.0%} deaths per batch; pos_weight set to {POS_WEIGHT:.1f}")
val_loader   = DataLoader(val_dataset, batch_size=min(CONFIG["BATCH_SIZE"], max(len(val_dataset), 1)),
                           shuffle=False, collate_fn=collate_bundle)
test_loader  = DataLoader(test_dataset, batch_size=min(CONFIG["BATCH_SIZE"], max(len(test_dataset), 1)),
                           shuffle=False, collate_fn=collate_bundle)

print(f"train_dataset: {len(train_dataset)} items "
      f"({train_dataset.n_real} real, {train_dataset.n_synthetic} synthetic, {len(AUGMENT_ITEMS)} augmented)")
print(f"val_dataset:   {len(val_dataset)} items")
print(f"test_dataset:  {len(test_dataset)} items")

_batch = next(iter(train_loader))
print("\nExample batch shapes:")
for k, v in _batch.items():
    if torch.is_tensor(v):
        print(f"  {k:20s} {tuple(v.shape)}")

# %% [markdown]
# ### 9.1b NEWS2 before vs. after sequence augmentation (v2)
#
# Jitter (σ = `JITTER_SIGMA` on the z-scored scale) and time-masking are applied to real
# died patients' sequences. **Before** = NEWS2 recomputed from each patient's original
# model grid; **after** = recomputed from the augmented grid, at the same time points
# (values converted back to clinical units first). A good augmentation perturbs values a
# little without routinely pushing patients into a different NEWS2 band.

# %%
def _grid_news2(sid, xs_by_system, masks_by_system):
    # NEWS2 per grid time point from a patient's (standardised) model tensors.
    vals = {}
    for system, item in [("respiratory", "rr"), ("respiratory", "spo2"), ("respiratory", "fio2"),
                         ("respiratory", "vent"), ("metabolic_hepatic", "bt"), ("cardiovascular", "nibp_sbp"),
                         ("cardiovascular", "hr"), ("neurological", "gcs_e"), ("neurological", "gcs_m"),
                         ("neurological", "gcs_v")]:
        fnames = PATIENT_BUNDLE[sid]["ts"][system]["feature_names"]
        if item not in fnames:
            vals[item] = np.full(CONFIG["TARGET_SEQ_LEN"], np.nan); continue
        j = fnames.index(item)
        x = xs_by_system[system][:, j] * TRAIN_STD[(system, item)] + STATS[(system, item)]["mean"]
        # round to charting precision (temp 0.1 C, everything else whole numbers): NEWS2 bands
        # are defined on charted values, so 95.95 vs 96 is not a real band change
        x = np.round(x, 1) if item == "bt" else np.round(x)
        vals[item] = np.where(masks_by_system[system][:, j] > 0, x, np.nan)   # observed points only
    comps = news2_components(rr=vals["rr"], spo2=vals["spo2"],
                             on_oxygen=on_supplemental_oxygen(vals["fio2"], vals["vent"], CONFIG["NEWS2_ASSUME_AIR_IF_MISSING"]),
                             sbp=vals["nibp_sbp"], hr=vals["hr"],
                             alert=alert_from_gcs(vals["gcs_e"], vals["gcs_m"], vals["gcs_v"]), temp_c=vals["bt"])
    total, _, any3 = news2_total(comps, CONFIG["NEWS2_ASSUME_ALERT_IF_MISSING"], CONFIG["NEWS2_MIN_CORE_VITALS"])
    return total, news2_risk_band(total, any3)

if not NEWS2_VALIDATED:
    print("NEWS2 self-test failed -- skipping the augmentation check.")
elif not AUGMENT_ITEMS:
    print("No sequence-augmented copies this run -- nothing to check.")
else:
    _b_tot, _a_tot, _b_band, _a_band, _n_masked, _n_points = [], [], [], [], 0, 0
    for sid, copy_idx in AUGMENT_ITEMS:
        rng = np.random.default_rng(_stable_seed(sid, copy_idx, 0))
        xs0 = {s: IMPUTED_BUNDLE[sid][s] for s in TIME_SERIES_SYSTEMS}
        ms0 = {s: PATIENT_BUNDLE[sid]["ts"][s]["mask"] for s in TIME_SERIES_SYSTEMS}
        xs1, ms1 = {}, {}
        for s in TIME_SERIES_SYSTEMS:   # same draw order as INSPIREDataset.__getitem__
            xs1[s], ms1[s] = _jitter_and_mask_sequence(xs0[s], ms0[s], CONFIG["JITTER_SIGMA"], rng, JITTER_COLS[s])
        t0, b0 = _grid_news2(sid, xs0, ms0)
        t1, b1 = _grid_news2(sid, xs1, ms1)
        both = ~np.isnan(t0) & ~np.isnan(t1)
        _n_points += int((~np.isnan(t0)).sum()); _n_masked += int((~np.isnan(t0) & np.isnan(t1)).sum())
        _b_tot.extend(t0[both]); _a_tot.extend(t1[both]); _b_band.extend(b0[both]); _a_band.extend(b1[both])
    _b_tot, _a_tot = np.array(_b_tot), np.array(_a_tot)
    res = compare_before_after(_b_tot, _a_tot, name="news2 per time point (jitter + time-mask)")
    res["mean_abs_change"] = float(np.mean(np.abs(_a_tot - _b_tot))) if len(_b_tot) else np.nan
    res["pct_band_changed"] = float(np.mean(np.array(_b_band) != np.array(_a_band))) if len(_b_band) else np.nan
    res["pct_points_hidden_by_time_mask"] = _n_masked / max(_n_points, 1)
    NEWS2_AUGMENTATION_CHECKS.append(res)
    print(pd.Series(res))
    print(f"\nReading this: jitter moved NEWS2 by {res['mean_abs_change']:.2f} points on average and changed "
          f"the risk band at {res['pct_band_changed']:.1%} of scorable time points. Under ~10% band changes "
          f"is a gentle augmentation; if it is much higher, lower CONFIG['JITTER_SIGMA'].")

# %% [markdown]
# ## 9.2–9.4 Model building blocks (v3)
#
# Inlined from `src-dnn/inspire_dnn/layers.py` (unit-tested there). In one line each:
#
# - **Eight organ systems, all built the same way.** Each owns its measurements over time
#   (none for GI and MSK in this dataset) and its own static features (its ICD-10 chapter,
#   its summary statistics — mapping printed in 9.6 for sign-off).
# - **`OrganSystemEncoder`** — ONE shared transformer body for all eight systems, a small
#   input adapter per system and a "which system am I" tag. ~8x fewer body parameters than a
#   network per system, and small systems borrow what data-rich systems teach the body.
# - **`WholePatientLayer`** — reads every feature at once. It supports each system network
#   by re-scaling that system's summary (0.5x–1.5x per channel: it can change how strongly a
#   system's own evidence counts, not add another system's values), and it has its own
#   whole-patient term in the risk. Starts as "no adjustment".
# - **`SystemCouplingLayer`** — the inter-system layer. Each system reads the other seven;
#   nothing hand-defined. **Pre-trained label-free** by hiding one system per patient and
#   predicting it from the other seven (so it learns from every patient, not just deaths),
#   then fine-tuned for mortality.
# - **`SystemArbitrationLayer`** — how much say each system's answer gets for this patient,
#   read from the whole-patient summary. Average say is 1; starts equal.
# - **`NAMFusion`** — risk logit = bias + whole-patient term + Σ say × answer. Exact.
#
# Guard-rails against memorising, all in `CONFIG`: shared body, label-free pre-training,
# gentle fine-tuning (encoder learning rate × 0.1), AdamW weight decay, organ-system
# dropout, capped SMOTENC, balanced batches. Part 11.2d (leakage check) and 11.4 (learning
# curve, train/val gap, ensemble stability) test whether they worked.

# %%
# INLINE: inspire_dnn/layers.py
from inspire_dnn.layers import *

# %% [markdown]
# ## 9.5 Loss functions — `pos_weight` BCE and focal loss (§1.4.1)
#
# Both take raw logits (no sigmoid applied beforehand) for numerical stability, matching
# `torch.nn.BCEWithLogitsLoss` conventions. `CONFIG['USE_FOCAL_LOSS']` switches between them
# independently of `CONFIG['SAMPLING_STRATEGY']` — the two are complementary, not
# mutually exclusive, per §1.4.1's note that focal loss "complements class weighting rather
# than replacing it."

# %%
def focal_loss_with_logits(logits, targets, pos_weight, gamma=2.0):
    # Lin et al. 2017 focal loss, generalised with a pos_weight term so it still respects
    # class imbalance (gamma alone down-weights EASY examples of either class; pos_weight
    # additionally up-weights the rarer class specifically -- combining both per §1.4.1).
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    weight = torch.where(targets == 1, torch.tensor(pos_weight, device=logits.device), torch.tensor(1.0, device=logits.device))
    return (weight * focal_term * bce).mean()

def mortality_loss(logits, targets, pos_weight, use_focal, gamma):
    if use_focal:
        return focal_loss_with_logits(logits, targets, pos_weight, gamma)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(pos_weight, device=logits.device))

print(f"Loss configured: {'focal loss' if CONFIG['USE_FOCAL_LOSS'] else 'pos_weight BCE'}, pos_weight={POS_WEIGHT:.2f}")

# %% [markdown]
# ## 9.6 Assembling the model — which static features belong to which organ system
#
# Every static column is assigned to exactly one organ system or to "patient context".
# The table is printed and saved (`organ_system_feature_map.csv`) for clinical sign-off —
# a few placements are judgement calls (e.g. vasopressor use under the heart). Context
# (age, ASA, department, NEWS2, frailty, infection flags, operation history, medication
# counts) belongs to no single system and is read by the whole-patient layer.

# %%
def assign_static_feature(name):
    if name.startswith("sysdx_"):
        for s in ORGAN_SYSTEMS:
            if name.startswith(f"sysdx_{s}_"):
                return s
    rules = [
        ("gi", ("gi_department_flag",)),
        ("msk", ("msk_department_flag",)),
        ("cardiovascular", ("cardio_", "agg_hr_", "agg_nibp_mbp_", "severe_hypotension", "vasopressor_")),
        ("renal", ("agg_creatinine_", "agg_potassium_")),
        ("respiratory", ("agg_spo2_",)),
        ("metabolic_hepatic", ("agg_glucose_", "agg_lacate_")),
        ("haematology", ("agg_wbc_", "agg_hb_")),
    ]
    for system, prefixes in rules:
        if name.startswith(prefixes):
            return system
    return "context"

STATIC_OWNER = {n: assign_static_feature(n) for n in STATIC_FEATURE_NAMES}
SYSTEM_STATIC_IDX = {s: [i for i, n in enumerate(STATIC_FEATURE_NAMES) if STATIC_OWNER[n] == s] for s in ORGAN_SYSTEMS}
CONTEXT_IDX = [i for i, n in enumerate(STATIC_FEATURE_NAMES) if STATIC_OWNER[n] == "context"]
TS_FEATURE_COUNTS = {s: (len(PATIENT_BUNDLE[TRAIN_IDS[0]]["ts"][s]["feature_names"]) if s in TIME_SERIES_SYSTEMS else 0)
                     for s in ORGAN_SYSTEMS}
SYSTEM_FEATURE_COUNTS = {s: TS_FEATURE_COUNTS[s] for s in TIME_SERIES_SYSTEMS}   # kept for older cells
N_STATIC_FEATURES = len(STATIC_FEATURE_NAMES)

organ_system_feature_map = pd.DataFrame([
    {"organ_system": s,
     "measurements_over_time": ", ".join(PATIENT_BUNDLE[TRAIN_IDS[0]]["ts"][s]["feature_names"]) if s in TIME_SERIES_SYSTEMS else "(none in this dataset)",
     "static_features": ", ".join(STATIC_FEATURE_NAMES[i] for i in SYSTEM_STATIC_IDX[s])}
    for s in ORGAN_SYSTEMS] + [{"organ_system": "patient context", "measurements_over_time": "-",
                                "static_features": ", ".join(STATIC_FEATURE_NAMES[i] for i in CONTEXT_IDX)}])
organ_system_feature_map.to_csv(os.path.join(CONFIG["CHECKPOINT_DIR"], "organ_system_feature_map.csv"), index=False)
for _, r in organ_system_feature_map.iterrows():
    print(f"{r['organ_system']:18s} | over time: {r['measurements_over_time'][:70]}")
    print(f"{'':18s} | static:    {r['static_features'][:110]}")

def build_model(fusion_strategy=None, coupling_mode=None, use_arbitration=None, use_whole_patient=None,
                share_encoder=None, seed=SEED):
    torch.manual_seed(seed)
    return MortalityModel(
        ts_feature_counts=TS_FEATURE_COUNTS, system_static_idx=SYSTEM_STATIC_IDX, context_idx=CONTEXT_IDX,
        n_static=N_STATIC_FEATURES,
        embed_dim=CONFIG["EMBED_DIM"], n_heads=CONFIG["TRANSFORMER_HEADS"], n_layers=CONFIG["TRANSFORMER_LAYERS"],
        ff_multiplier=CONFIG["FF_MULTIPLIER"], target_seq_len=CONFIG["TARGET_SEQ_LEN"],
        fusion_strategy=fusion_strategy or CONFIG["FUSION_STRATEGY"],
        share_encoder=CONFIG["SHARE_ENCODER"] if share_encoder is None else share_encoder,
        coupling_mode=coupling_mode or CONFIG["COUPLING_MODE"], coupling_heads=CONFIG["COUPLING_HEADS"],
        coupling_prior_links=[tuple(l) for l in CONFIG["COUPLING_PRIOR_LINKS"]],
        coupling_prior_strength=CONFIG["COUPLING_PRIOR_STRENGTH"], coupling_gate_init=CONFIG["COUPLING_GATE_INIT"],
        use_arbitration=CONFIG["USE_ARBITRATION_LAYER"] if use_arbitration is None else use_arbitration,
        use_whole_patient=CONFIG["USE_WHOLE_PATIENT_LAYER"] if use_whole_patient is None else use_whole_patient,
        system_dropout=CONFIG["SYSTEM_DROPOUT"], dropout=CONFIG["DROPOUT"],
    ).to(DEVICE)

model = build_model()

def _count(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0
n_params = _count(model)
_n_enc = _count(model.encoder)
print(f"\nMortalityModel built: {n_params:,} parameters in total")
print(f"  organ-system encoder ({'one shared body' if CONFIG['SHARE_ENCODER'] else 'a body per system'}): {_n_enc:,}")
print(f"  whole-patient layer: {_count(model.whole_patient):,}   coupling layer: {_count(model.coupling):,}   "
      f"arbitration layer: {_count(model.arbitration):,}")
print(f"Terms in the risk (each reported separately): {model.term_names}")

import hashlib
_fingerprint_source = repr((
    TS_FEATURE_COUNTS, N_STATIC_FEATURES, SYSTEM_STATIC_IDX, CONFIG["EMBED_DIM"], CONFIG["TRANSFORMER_HEADS"],
    CONFIG["TRANSFORMER_LAYERS"], CONFIG["FF_MULTIPLIER"], CONFIG["TARGET_SEQ_LEN"], CONFIG["FUSION_STRATEGY"],
    CONFIG["SHARE_ENCODER"], CONFIG["COUPLING_MODE"], CONFIG["COUPLING_HEADS"], CONFIG["USE_ARBITRATION_LAYER"],
    CONFIG["USE_WHOLE_PATIENT_LAYER"], "v3",
))
ARCHITECTURE_FINGERPRINT = hashlib.sha256(_fingerprint_source.encode()).hexdigest()[:10]
print(f"Architecture fingerprint: {ARCHITECTURE_FINGERPRINT} (checkpoints are tagged with this)")

_batch_dev = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in _batch.items()}
model.eval()
with torch.no_grad():
    _out = model(_batch_dev)
print(f"\nSmoke test forward pass OK. logit shape: {tuple(_out['logit'].shape)}")
if _out["per_system_signal"] is not None:
    _recon = sum(_out["per_system_signal"].values()) + model.fusion.bias
    print(f"Terms sum exactly to the logit: {torch.allclose(_recon, _out['logit'], atol=1e-4)}")
