"""
Stage 3: missing data -- the four imputation strategies (decision-tree default, median,
interpolate, KNN), standardization (z-score, fit on train only), and the
leakage-safe stats/split used for both. Converted from cells 59-80, preserved verbatim
(no logic changes in this stage). Confirmed column-agnostic for static features
(impute_static_vector fills NaN generically per-column), so the 5 new NEWS2 static
features added in news2_integration.py flow through this stage with no changes needed.

NOTE: this stage includes two genuinely expensive, OPTIONAL benchmark sections
(7.6 held-out imputation-accuracy benchmark, 7.7 regression/MICE imputation) --
CONFIG["SKIP_IMPUTATION_ACCURACY_BENCHMARKS"] (default True) skips both, same as the
original notebook's own time-budget control.
"""
from news2_integration import *

# --- from notebook cell 61 ---
def population_stats(train_ids):
    # Per (system, feature) mean/median across the TRAINING split only -- used whenever a
    # patient has zero observations of a feature anywhere in their window (§1.3.2's first
    # branch: nothing else is possible in that case).
    stats = {}
    for system in TIME_SERIES_SYSTEMS:
        fnames = PATIENT_BUNDLE[train_ids[0]]["ts"][system]["feature_names"]
        if not fnames:
            continue
        all_vals = []
        for sid in train_ids:
            raw = PATIENT_BUNDLE[sid]["ts"][system]["raw"]
            all_vals.append(raw)
        stacked = np.concatenate(all_vals, axis=0) if all_vals else np.zeros((0, len(fnames)))
        for j, fname in enumerate(fnames):
            col = stacked[:, j]
            col = col[~np.isnan(col)]
            stats[(system, fname)] = {
                "mean": float(np.mean(col)) if len(col) else 0.0,
                "median": float(np.median(col)) if len(col) else 0.0,
            }
    return stats

# --- from notebook cell 63 ---
def impute_median(raw, feature_names, system, stats):
    # §1.3.1 row 1/2: fill every NaN with the training-set median for that feature.
    out = raw.copy()
    for j, fname in enumerate(feature_names):
        med = stats[(system, fname)]["median"]
        col = out[:, j]
        col[np.isnan(col)] = med
    return out

def _fade_to_mean_interpolate(times_observed, values_observed, grid, mean_val, decay=8.0):
    # The source repo's smooth_fade_to_mean_interpolator idea (§1.3.1 row 3), reimplemented
    # standalone: linear interpolation between observed points, fading toward the population
    # mean the further a grid point is from any observation.
    if len(times_observed) == 0:
        return np.full(len(grid), mean_val)
    if len(times_observed) == 1:
        # can't interpolate a slope from one point -- fall back to nearest + fade
        t0, v0 = times_observed[0], values_observed[0]
        out = np.array([v0 * math.exp(-abs(g - t0) / decay) + mean_val * (1 - math.exp(-abs(g - t0) / decay)) for g in grid])
        return out
    interp = np.interp(grid, times_observed, values_observed)
    t_min, t_max = times_observed.min(), times_observed.max()
    out = interp.copy()
    for i, g in enumerate(grid):
        if g < t_min:
            w = math.exp(-abs(t_min - g) / decay)
            out[i] = w * interp[i] + (1 - w) * mean_val
        elif g > t_max:
            w = math.exp(-abs(g - t_max) / decay)
            out[i] = w * interp[i] + (1 - w) * mean_val
    return out

def impute_interpolate(raw, feature_names, system, stats, grid):
    # §1.3.1 row 3: within-patient interpolation, fading to the population mean outside
    # the observed range -- the default for fast-changing vitals per §1.3.2's decision tree.
    out = raw.copy()
    n_t = raw.shape[0]
    for j, fname in enumerate(feature_names):
        col = out[:, j]
        obs_idx = ~np.isnan(col)
        mean_val = stats[(system, fname)]["mean"]
        if obs_idx.sum() == 0:
            out[:, j] = mean_val
        else:
            out[:, j] = _fade_to_mean_interpolate(grid[obs_idx], col[obs_idx], grid, mean_val)
    return out

def impute_forward_fill(raw, feature_names, system, stats):
    # §1.3.1 row 4 (LOCF): repeat the last observed value forward; before any observation,
    # fall back to the training-set median (there's nothing to carry forward from yet).
    out = raw.copy()
    n_t = raw.shape[0]
    for j in range(out.shape[1]):
        last_val = None
        med = stats[(system, feature_names[j])]["median"]
        for t in range(n_t):
            if np.isnan(out[t, j]):
                out[t, j] = last_val if last_val is not None else med
            else:
                last_val = out[t, j]
    return out

def impute_knn(raw_matrix_all_patients, k):
    # §1.3.1 row 5: KNN across patients, one call per (system) on a [n_patients, T*F]
    # flattened matrix -- used at the whole-cohort level in §7.3, not per-patient like the
    # other three, since KNN needs other patients to find neighbours from.
    imputer = KNNImputer(n_neighbors=min(k, max(raw_matrix_all_patients.shape[0] - 1, 1)))
    return imputer.fit_transform(raw_matrix_all_patients)

# --- from notebook cell 65 ---
FAST_CHANGING_ITEMS = {
    "hr", "nibp_sbp", "nibp_dbp", "nibp_mbp", "spo2", "rr", "fio2", "bt", "uo",
    "art_sbp", "art_dbp", "art_mbp", "etco2", "peep", "pip", "pplat", "gcs_e", "gcs_m", "gcs_v",
}
# Everything else in the routing map (labs like creatinine, albumin, wbc, ...) is treated
# as slow-changing -> forward-fill, per §1.3.2.

def impute_patient_system(sid, system, strategy, stats):
    bundle = PATIENT_BUNDLE[sid]["ts"][system]
    raw, fnames = bundle["raw"], bundle["feature_names"]
    if raw.shape[1] == 0:
        return raw.copy()
    row = COHORT_INDEXED.loc[sid]
    lo, hi = get_window(sid, row)
    grid = np.linspace(lo, hi, CONFIG["TARGET_SEQ_LEN"])

    if strategy == "median":
        return impute_median(raw, fnames, system, stats)
    elif strategy == "interpolate":
        return impute_interpolate(raw, fnames, system, stats, grid)
    elif strategy == "decision_tree":
        out = raw.copy()
        fast_idx = [j for j, f in enumerate(fnames) if f in FAST_CHANGING_ITEMS]
        slow_idx = [j for j, f in enumerate(fnames) if f not in FAST_CHANGING_ITEMS]
        if fast_idx:
            fast_names = [fnames[j] for j in fast_idx]
            out[:, fast_idx] = impute_interpolate(raw[:, fast_idx], fast_names, system, stats, grid)
        if slow_idx:
            slow_names = [fnames[j] for j in slow_idx]
            out[:, slow_idx] = impute_forward_fill(raw[:, slow_idx], slow_names, system, stats)
        return out
    else:
        raise ValueError(f"strategy {strategy!r} handled elsewhere (knn) or unknown")

def impute_static_vector(sid, train_static_df):
    # Static/tabular features (§6.14): median-fill per column, fit on the training split
    # only (passed in) -- same leakage discipline as the time-series imputers above.
    vec = pd.Series(PATIENT_BUNDLE[sid]["static"])
    filled = vec.fillna(train_static_df.median(numeric_only=True))
    return filled

# --- from notebook cell 67 ---
# A first, lightweight split done here (before §8's full split/sampling section) purely to
# get TRAIN_IDS for leakage-safe statistics -- §8 re-derives and USES this same split for
# training; nothing here duplicates or conflicts with it.
ALL_IDS = cohort_df["subject_id"].tolist()
ALL_LABELS = np.array([PATIENT_BUNDLE[sid]["label"] for sid in ALL_IDS])
TRAIN_IDS_FOR_STATS, _HOLD_IDS = train_test_split(
    ALL_IDS, test_size=(CONFIG["VAL_FRACTION"] + CONFIG["TEST_FRACTION"]),
    stratify=ALL_LABELS, random_state=SEED
)

STATS = population_stats(TRAIN_IDS_FOR_STATS)
train_static_df = pd.DataFrame({sid: PATIENT_BUNDLE[sid]["static"] for sid in TRAIN_IDS_FOR_STATS}).T

IMPUTED_BUNDLE = {}
strategy = CONFIG["IMPUTATION_STRATEGY"]

if strategy == "knn":
    # Build one big matrix per system: [n_patients, T*F], impute across patients, reshape back.
    for system in TIME_SERIES_SYSTEMS:
        fnames = PATIENT_BUNDLE[ALL_IDS[0]]["ts"][system]["feature_names"]
        if not fnames:
            continue
        mat = np.stack([PATIENT_BUNDLE[sid]["ts"][system]["raw"].reshape(-1) for sid in ALL_IDS])
        mat_imputed = impute_knn(mat, CONFIG["KNN_NEIGHBORS"])
        for i, sid in enumerate(ALL_IDS):
            IMPUTED_BUNDLE.setdefault(sid, {})[system] = mat_imputed[i].reshape(CONFIG["TARGET_SEQ_LEN"], len(fnames))
else:
    for sid in ALL_IDS:
        IMPUTED_BUNDLE[sid] = {}
        for system in TIME_SERIES_SYSTEMS:
            IMPUTED_BUNDLE[sid][system] = impute_patient_system(sid, system, strategy, STATS)

for sid in ALL_IDS:
    IMPUTED_BUNDLE[sid]["static"] = impute_static_vector(sid, train_static_df)

print(f"Imputation strategy used: {strategy!r}")
remaining_nans = sum(np.isnan(IMPUTED_BUNDLE[sid][system]).sum()
                      for sid in ALL_IDS for system in TIME_SERIES_SYSTEMS
                      if IMPUTED_BUNDLE[sid][system].size)
print(f"Remaining NaNs after imputation (should be 0): {remaining_nans}")

# --- from notebook cell 69 ---
# Pick one patient/system/feature with a genuine, visible gap and show what each strategy
# does to it -- exactly the kind of comparison you said you want to be able to run yourself.
example_sid = ALL_IDS[0]
example_system = "renal"
example_feature_idx = 0
row = COHORT_INDEXED.loc[example_sid]
lo, hi = get_window(example_sid, row)
grid = np.linspace(lo, hi, CONFIG["TARGET_SEQ_LEN"])
raw = PATIENT_BUNDLE[example_sid]["ts"][example_system]["raw"]
fnames = PATIENT_BUNDLE[example_sid]["ts"][example_system]["feature_names"]

fig, ax = plt.subplots(figsize=(9, 4.5))
strategies_to_show = ["median", "interpolate", "decision_tree"]
colors = {"median": "#e67e22", "interpolate": "#2980b9", "decision_tree": "#27ae60"}
for strat in strategies_to_show:
    filled = impute_patient_system(example_sid, example_system, strat, STATS)
    ax.plot(grid, filled[:, example_feature_idx], label=strat, color=colors[strat], alpha=0.8)
obs_mask = ~np.isnan(raw[:, example_feature_idx])
ax.scatter(grid[obs_mask], raw[obs_mask, example_feature_idx], color="black", zorder=5, label="observed", s=40)
ax.set_title(f"Imputation comparison: patient {example_sid}, {example_system}/{fnames[example_feature_idx]}")
ax.set_xlabel("chart_time (minutes)"); ax.set_ylabel("value"); ax.legend()
plt.tight_layout(); plt.show()
print("This is exactly the comparison to re-run (swap example_sid/example_system/example_feature_idx) "
      "as you experiment with §1.3's strategies on your own features of interest.")

# --- from notebook cell 71 ---
def imputation_accuracy_benchmark(systems, patient_ids, mask_fraction=0.2, seed=SEED, max_patients=300):
    rng = np.random.default_rng(seed)
    bench_ids = list(patient_ids)
    if len(bench_ids) > max_patients:
        bench_ids = list(rng.choice(bench_ids, size=max_patients, replace=False))

    rows = []
    for system in TIME_SERIES_SYSTEMS:
        fnames = PATIENT_BUNDLE[bench_ids[0]]["ts"][system]["feature_names"]
        if not fnames:
            continue
        for strategy in ["median", "interpolate", "decision_tree"]:
            all_true, all_pred = [], []
            for sid in bench_ids:
                row_info = COHORT_INDEXED.loc[sid]
                lo, hi = get_window(sid, row_info)
                grid = np.linspace(lo, hi, CONFIG["TARGET_SEQ_LEN"])
                raw_orig = PATIENT_BUNDLE[sid]["ts"][system]["raw"]
                mask_orig = PATIENT_BUNDLE[sid]["ts"][system]["mask"]
                obs_positions = np.argwhere(mask_orig == 1)
                if len(obs_positions) == 0:
                    continue
                n_mask = max(1, int(len(obs_positions) * mask_fraction))
                n_mask = min(n_mask, len(obs_positions))
                chosen_idx = rng.choice(len(obs_positions), size=n_mask, replace=False)
                chosen = obs_positions[chosen_idx]

                raw_masked = raw_orig.copy()
                true_vals = []
                for (t_idx, f_idx) in chosen:
                    true_vals.append(raw_orig[t_idx, f_idx])
                    raw_masked[t_idx, f_idx] = np.nan

                if strategy == "median":
                    imputed = impute_median(raw_masked, fnames, system, STATS)
                elif strategy == "interpolate":
                    imputed = impute_interpolate(raw_masked, fnames, system, STATS, grid)
                else:  # decision_tree
                    imputed = raw_masked.copy()
                    fast_idx = [j for j, f in enumerate(fnames) if f in FAST_CHANGING_ITEMS]
                    slow_idx = [j for j, f in enumerate(fnames) if f not in FAST_CHANGING_ITEMS]
                    if fast_idx:
                        imputed[:, fast_idx] = impute_interpolate(raw_masked[:, fast_idx], [fnames[j] for j in fast_idx], system, STATS, grid)
                    if slow_idx:
                        imputed[:, slow_idx] = impute_forward_fill(raw_masked[:, slow_idx], [fnames[j] for j in slow_idx], system, STATS)

                for (t_idx, f_idx), true_v in zip(chosen, true_vals):
                    all_true.append(true_v)
                    all_pred.append(imputed[t_idx, f_idx])

            if all_true:
                all_true_arr, all_pred_arr = np.array(all_true), np.array(all_pred)
                mae = float(np.mean(np.abs(all_pred_arr - all_true_arr)))
                rows.append({"system": system, "strategy": strategy, "mae": mae, "n_masked_points": len(all_true)})

    return pd.DataFrame(rows)

if CONFIG["SKIP_IMPUTATION_ACCURACY_BENCHMARKS"]:
    print("CONFIG['SKIP_IMPUTATION_ACCURACY_BENCHMARKS']=True -- skipping this benchmark to "
          "keep total runtime inside the 20-30 minute budget. The function above is still "
          "defined -- set the flag to False and re-run this cell to get real numbers.")
    IMPUTATION_BENCHMARK = pd.DataFrame(columns=["system", "strategy", "mae", "n_masked_points"])
else:
    IMPUTATION_BENCHMARK = imputation_accuracy_benchmark(TIME_SERIES_SYSTEMS, TRAIN_IDS_FOR_STATS)
    print(f"Benchmark built on {IMPUTATION_BENCHMARK['n_masked_points'].sum() if len(IMPUTATION_BENCHMARK) else 0} "
          f"held-out (masked) real data points across all systems.")
IMPUTATION_BENCHMARK

# --- from notebook cell 72 ---
# Small multiples -- one subplot per system, each on its own scale (a shared axis across
# systems would make e.g. glucose's MAE look "worse" than creatinine's purely because of
# unit scale, not because the method is worse -- see the note above this section).
_systems_with_data = IMPUTATION_BENCHMARK["system"].unique().tolist() if len(IMPUTATION_BENCHMARK) else []
if _systems_with_data:
    n_sys = len(_systems_with_data)
    fig, axes = plt.subplots(1, n_sys, figsize=(3.2 * n_sys, 4), sharey=False)
    if n_sys == 1:
        axes = [axes]
    strategy_colors = {"median": "#95a5a6", "interpolate": "#2980b9", "decision_tree": "#27ae60"}
    for ax, system in zip(axes, _systems_with_data):
        sub = IMPUTATION_BENCHMARK[IMPUTATION_BENCHMARK["system"] == system].set_index("strategy")
        strategies = ["median", "interpolate", "decision_tree"]
        maes = [sub.loc[s, "mae"] if s in sub.index else 0 for s in strategies]
        colors = [strategy_colors[s] for s in strategies]
        ax.bar(strategies, maes, color=colors)
        best = strategies[int(np.argmin(maes))] if any(m > 0 for m in maes) else None
        ax.set_title(system.replace("_", "\n"), fontsize=10)
        ax.set_ylabel("mean absolute error\n(original clinical units)")
        ax.tick_params(axis="x", rotation=30)
        if best:
            ax.set_xlabel(f"best: {best}", fontsize=9, color="#27ae60")
    plt.suptitle("Imputation accuracy by system -- lower is better (held-out masking benchmark, §7.6)")
    plt.tight_layout()
    plt.show()

    print("\nBest strategy per system (lowest held-out error):")
    for system in _systems_with_data:
        sub = IMPUTATION_BENCHMARK[IMPUTATION_BENCHMARK["system"] == system]
        best_row = sub.loc[sub["mae"].idxmin()]
        print(f"  {system:18s} -> {best_row['strategy']:14s} (MAE={best_row['mae']:.3f}, n={int(best_row['n_masked_points'])} held-out points)")
else:
    if CONFIG["SKIP_IMPUTATION_ACCURACY_BENCHMARKS"]:
        print("Skipped (CONFIG['SKIP_IMPUTATION_ACCURACY_BENCHMARKS']=True) -- not a data problem.")
    else:
        print("No systems had enough observed data to benchmark on this sample -- re-run at larger scale.")

# --- from notebook cell 74 ---
def build_regression_imputer(system, train_ids, fnames):
    # Pools every (patient, timestep) row across the given patients into one big matrix
    # -- IterativeImputer needs real cross-patient volume to learn feature relationships,
    # which a single patient's ~24 timesteps could never provide on their own.
    rows = []
    for sid in train_ids:
        raw = PATIENT_BUNDLE[sid]["ts"][system]["raw"]
        for t in range(raw.shape[0]):
            rows.append(raw[t, :])
    X = np.array(rows)
    imputer = IterativeImputer(max_iter=10, random_state=SEED, sample_posterior=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # IterativeImputer is verbose about near-constant/all-NaN columns on sparse data -- expected here, not a bug
        imputer.fit(X)
    return imputer

def impute_regression(raw, imputer):
    # raw: [T, F] for ONE patient, with NaN gaps (including any artificially held-out
    # points from the benchmark below). Uses the population-fit imputer from above.
    with warnings.catch_warnings():
        simplefilter = warnings.simplefilter
        simplefilter("ignore")
        return imputer.transform(raw)

print("Regression (MICE) imputation defined -- benchmarked against the rule-based methods below.")

# --- from notebook cell 76 ---
def imputation_accuracy_benchmark_with_regression(systems, patient_ids, mask_fraction=0.2, seed=SEED, max_patients=300):
    rng = np.random.default_rng(seed)
    bench_ids = list(patient_ids)
    if len(bench_ids) > max_patients:
        bench_ids = list(rng.choice(bench_ids, size=max_patients, replace=False))

    rows = []
    for system in TIME_SERIES_SYSTEMS:
        fnames = PATIENT_BUNDLE[bench_ids[0]]["ts"][system]["feature_names"]
        if not fnames or len(fnames) < 2:   # regression imputation needs >=2 features to have anything to regress from
            continue

        # Decide the held-out mask ONCE per system, shared across all four methods, so
        # this is a fair identical-conditions comparison, not different gaps per method.
        masked_raw_by_patient, true_by_patient = {}, {}
        for sid in bench_ids:
            raw_orig = PATIENT_BUNDLE[sid]["ts"][system]["raw"]
            mask_orig = PATIENT_BUNDLE[sid]["ts"][system]["mask"]
            obs_positions = np.argwhere(mask_orig == 1)
            if len(obs_positions) == 0:
                continue
            n_mask = max(1, int(len(obs_positions) * mask_fraction))
            n_mask = min(n_mask, len(obs_positions))
            chosen = obs_positions[rng.choice(len(obs_positions), size=n_mask, replace=False)]
            raw_masked = raw_orig.copy()
            true_vals = {}
            for (t_idx, f_idx) in chosen:
                true_vals[(t_idx, f_idx)] = raw_orig[t_idx, f_idx]
                raw_masked[t_idx, f_idx] = np.nan
            masked_raw_by_patient[sid] = raw_masked
            true_by_patient[sid] = true_vals

        if not masked_raw_by_patient:
            continue

        # Fit the regression imputer on the SAME masked training data (not leaking the
        # held-out true values into its own fit).
        try:
            reg_imputer = build_regression_imputer(system, list(masked_raw_by_patient.keys()), fnames)
        except Exception as e:
            reg_imputer = None
            print(f"  {system}: regression imputer failed to fit ({e}) -- skipping regression for this system")

        for strategy in ["median", "interpolate", "decision_tree", "regression"]:
            if strategy == "regression" and reg_imputer is None:
                continue
            all_true, all_pred = [], []
            for sid, raw_masked in masked_raw_by_patient.items():
                row_info = COHORT_INDEXED.loc[sid]
                lo, hi = get_window(sid, row_info)
                grid = np.linspace(lo, hi, CONFIG["TARGET_SEQ_LEN"])

                if strategy == "median":
                    imputed = impute_median(raw_masked, fnames, system, STATS)
                elif strategy == "interpolate":
                    imputed = impute_interpolate(raw_masked, fnames, system, STATS, grid)
                elif strategy == "decision_tree":
                    imputed = raw_masked.copy()
                    fast_idx = [j for j, f in enumerate(fnames) if f in FAST_CHANGING_ITEMS]
                    slow_idx = [j for j, f in enumerate(fnames) if f not in FAST_CHANGING_ITEMS]
                    if fast_idx:
                        imputed[:, fast_idx] = impute_interpolate(raw_masked[:, fast_idx], [fnames[j] for j in fast_idx], system, STATS, grid)
                    if slow_idx:
                        imputed[:, slow_idx] = impute_forward_fill(raw_masked[:, slow_idx], [fnames[j] for j in slow_idx], system, STATS)
                else:  # regression
                    imputed = impute_regression(raw_masked, reg_imputer)

                for (t_idx, f_idx), true_v in true_by_patient[sid].items():
                    all_true.append(true_v)
                    all_pred.append(imputed[t_idx, f_idx])

            if all_true:
                mae = float(np.mean(np.abs(np.array(all_pred) - np.array(all_true))))
                rows.append({"system": system, "strategy": strategy, "mae": mae, "n_masked_points": len(all_true)})

    return pd.DataFrame(rows)

if CONFIG["SKIP_IMPUTATION_ACCURACY_BENCHMARKS"]:
    print("CONFIG['SKIP_IMPUTATION_ACCURACY_BENCHMARKS']=True -- skipping this benchmark. "
          "This is the single most expensive optional section in the notebook (IterativeImputer/"
          "MICE fitting is iterative) -- set the flag to False and re-run once you've confirmed "
          "your session's timing headroom.")
    IMPUTATION_BENCHMARK_V2 = pd.DataFrame(columns=["system", "strategy", "mae", "n_masked_points"])
else:
    IMPUTATION_BENCHMARK_V2 = imputation_accuracy_benchmark_with_regression(TIME_SERIES_SYSTEMS, TRAIN_IDS_FOR_STATS)
IMPUTATION_BENCHMARK_V2

# --- from notebook cell 77 ---
_systems_v2 = IMPUTATION_BENCHMARK_V2["system"].unique().tolist() if len(IMPUTATION_BENCHMARK_V2) else []
if _systems_v2:
    n_sys = len(_systems_v2)
    fig, axes = plt.subplots(1, n_sys, figsize=(3.4 * n_sys, 4), sharey=False)
    if n_sys == 1:
        axes = [axes]
    strategy_colors = {"median": "#95a5a6", "interpolate": "#2980b9", "decision_tree": "#27ae60", "regression": "#e74c3c"}
    strategies_order = ["median", "interpolate", "decision_tree", "regression"]
    for ax, system in zip(axes, _systems_v2):
        sub = IMPUTATION_BENCHMARK_V2[IMPUTATION_BENCHMARK_V2["system"] == system].set_index("strategy")
        present = [s for s in strategies_order if s in sub.index]
        maes = [sub.loc[s, "mae"] for s in present]
        colors = [strategy_colors[s] for s in present]
        ax.bar(present, maes, color=colors)
        best = present[int(np.argmin(maes))] if maes else None
        ax.set_title(system.replace("_", "\n"), fontsize=10)
        ax.set_ylabel("MAE (clinical units)")
        ax.tick_params(axis="x", rotation=35)
        if best:
            ax.set_xlabel(f"best: {best}", fontsize=9, color="#27ae60")
    plt.suptitle("Imputation accuracy, rule-based methods vs. regression (MICE) -- lower is better (§7.7)")
    plt.tight_layout()
    plt.show()

    n_regression_wins = 0
    for system in _systems_v2:
        sub = IMPUTATION_BENCHMARK_V2[IMPUTATION_BENCHMARK_V2["system"] == system]
        best_row = sub.loc[sub["mae"].idxmin()]
        if best_row["strategy"] == "regression":
            n_regression_wins += 1
        print(f"  {system:18s} -> best: {best_row['strategy']:14s} (MAE={best_row['mae']:.3f})")
    print(f"\nRegression imputation won on {n_regression_wins}/{len(_systems_v2)} systems on this sample.")
    print("Re-run this once trained on the larger cohort -- MICE specifically benefits from more "
          "rows to learn cross-feature relationships from, more than the rule-based methods do.")
else:
    if CONFIG["SKIP_IMPUTATION_ACCURACY_BENCHMARKS"]:
        print("Skipped (CONFIG['SKIP_IMPUTATION_ACCURACY_BENCHMARKS']=True) -- not a data problem.")
    else:
        print("No systems had enough features/data to benchmark regression imputation on this sample.")

# --- from notebook cell 79 ---
def compute_train_std(train_ids):
    std_stats = {}
    for system in TIME_SERIES_SYSTEMS:
        fnames = PATIENT_BUNDLE[train_ids[0]]["ts"][system]["feature_names"]
        if not fnames:
            continue
        stacked = np.stack([IMPUTED_BUNDLE[sid][system] for sid in train_ids])   # [N, T, F]
        for j, fname in enumerate(fnames):
            col = stacked[:, :, j].reshape(-1)
            std_stats[(system, fname)] = float(col.std() + 1e-6)
    return std_stats

TRAIN_STD = compute_train_std(TRAIN_IDS_FOR_STATS)

def standardize_system_tensor(raw, feature_names, system):
    out = raw.copy()
    for j, fname in enumerate(feature_names):
        mean = STATS[(system, fname)]["mean"]
        std = TRAIN_STD[(system, fname)]
        out[:, j] = (out[:, j] - mean) / std
    return out

for sid in ALL_IDS:
    for system in TIME_SERIES_SYSTEMS:
        fnames = PATIENT_BUNDLE[sid]["ts"][system]["feature_names"]
        if fnames:
            IMPUTED_BUNDLE[sid][system] = standardize_system_tensor(IMPUTED_BUNDLE[sid][system], fnames, system)

# Static features: same idea, via sklearn's StandardScaler fit on the training static table.
static_scaler = StandardScaler()
static_scaler.fit(train_static_df.fillna(train_static_df.median(numeric_only=True)).values)

for sid in ALL_IDS:
    vec = IMPUTED_BUNDLE[sid]["static"]
    scaled = static_scaler.transform(vec.values.reshape(1, -1))[0]
    IMPUTED_BUNDLE[sid]["static"] = pd.Series(scaled, index=vec.index)

print("Standardisation applied (train-fit z-score) to all time-series and static features.")
example_sid = ALL_IDS[0]
print(f"Example, patient {example_sid}, renal system, post-standardisation value range: "
      f"[{IMPUTED_BUNDLE[example_sid]['renal'].min():.2f}, {IMPUTED_BUNDLE[example_sid]['renal'].max():.2f}] "
      f"(should now be roughly within a few units of 0, not raw clinical units)")

# --- from notebook cell 80 ---
record_checkpoint("Part 7 -- missing-data imputation + standardisation")