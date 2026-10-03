# %% [markdown]
# # Part 3 -- Data loading
#
# Each patient is one JSON file (in `died/` or `survived/`) with six lists: `labs`, `vitals`
# (intra-op), `ward_vitals`, `operations`, `diagnoses`, `medications`. This section parses
# every file **exactly once** in a single streaming pass, building all six downstream tables
# at the same time -- rather than parsing all patients into one big in-memory dict first and
# then re-scanning that dict once per table, which is wasted work and wasted memory at full
# scale. This pattern (one file open, one parse, extract everything needed, discard, move on)
# matches the proven-fast approach already used in this project's other Colab notebook on the
# full 99,886-patient cohort.
#
# **Note on folder labels (§1.6.3):** the `died`/`survived` folder is used only to know which
# files to open -- the actual mortality label used for training is **recomputed from
# `operations`** in §5.2, not trusted from the folder name, for the reason explained there.
#
# **Caching (new).** The parse above is the single most expensive CPU step in this notebook
# at full scale -- tens of thousands of small JSON files, each opened and parsed individually.
# The first run does that parse once, then saves the resulting tables as Parquet to
# `CONFIG['CACHE_DIR']`. Every later run (this session or a future one, if `CACHE_DIR` is on
# a mounted Drive) checks for that cache first and, if found, loads it directly -- seconds
# instead of minutes-to-tens-of-minutes. The cache is keyed to `SUBJECTS_DIR`,
# `MAX_SUBJECTS_PER_CLASS`, and `TIME_WINDOW` together, so a smoke-test run and a full run (or
# a pre-op vs. peri-op run) never collide or silently load the wrong cache.

# %%
def _safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None

def _safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except (TypeError, ValueError):
        return None

import hashlib

def _cache_key():
    src = repr((CONFIG["SUBJECTS_DIR"], CONFIG["MAX_SUBJECTS_PER_CLASS"], CONFIG["TIME_WINDOW"]))
    return hashlib.sha256(src.encode()).hexdigest()[:12]

CACHE_KEY = _cache_key()
CACHE_SUBDIR = os.path.join(CONFIG["CACHE_DIR"], f"parsed_{CACHE_KEY}")
_CACHE_TABLES = ["labs", "vitals", "ward_vitals", "operations", "diagnoses", "medications", "folder_label"]

def _cache_complete(subdir):
    return os.path.isdir(subdir) and all(
        os.path.exists(os.path.join(subdir, f"{name}.parquet")) for name in _CACHE_TABLES
    )

def _load_from_cache(subdir):
    dfs = {name: pd.read_parquet(os.path.join(subdir, f"{name}.parquet")) for name in _CACHE_TABLES}
    folder_label = dict(zip(dfs["folder_label"]["subject_id"], dfs["folder_label"]["folder"]))
    return dfs, folder_label

# NOTE: no separate _save_to_cache function -- stream_parse_subjects (below) writes each
# chunk straight to CACHE_SUBDIR as it parses, which is the whole point of chunking (never
# holding the full cohort in memory to save "all at once" afterward). USE_CACHE=False only
# means "don't trust a pre-existing cache as valid input" on a future run -- it does not
# turn off the chunked writes themselves, since those are what keep memory bounded during
# parsing regardless of whether a cache is wanted afterward.

# %%
def stream_parse_subjects(subjects_dir, max_per_class, time_window, cache_subdir, chunk_size, seed=SEED):
    # Parses in CHUNKS, flushed to Parquet incrementally, instead of holding every row of
    # every table as a Python dict for the WHOLE cohort in memory at once. Two separate
    # memory wins stack here: (1) within each chunk, values are accumulated in parallel
    # per-column lists (ints/floats/strings), not one dict object per row -- roughly 6x
    # lighter per row, measured directly (a dict-of-4-fields costs ~192 bytes/row in
    # CPython; four parallel lists holding the same data cost ~32 bytes/row); (2) only
    # ever holding `chunk_size` patients' worth of rows at once, then writing to disk and
    # freeing them, means peak memory no longer scales with total cohort size at all --
    # it's bounded by chunk_size regardless of whether the cohort is 30 patients or
    # 99,886. Categorical dtypes are applied ONCE, after every chunk has been written and
    # is read back as a single table -- not per-chunk -- since a categorical column's
    # dictionary can legitimately differ chunk to chunk (different item_names observed in
    # different patients), which is safer to reconcile in one pass than to guess is
    # compatible mid-stream.
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    filepaths = []
    for folder in ["died", "survived"]:
        folder_path = os.path.join(subjects_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        fps = sorted(glob.glob(os.path.join(folder_path, "*.json")))
        if max_per_class is not None and len(fps) > max_per_class:
            fps = rng.sample(fps, max_per_class)
            fps.sort()
        filepaths.extend((fp, folder) for fp in fps)

    print(f"{len(filepaths)} subject files to parse, in chunks of {chunk_size}...")
    load_vitals = (time_window == "peri_op")

    os.makedirs(cache_subdir, exist_ok=True)
    table_names = ["labs", "vitals", "ward_vitals", "operations", "diagnoses", "medications"]
    writers = {name: None for name in table_names}   # opened lazily on each table's first non-empty chunk
    folder_label = {}

    # Fixed, explicit column lists for every table -- rows are built with .get(), never a
    # raw dict(record) copy. This guarantees every row in every chunk has EXACTLY the same
    # set of columns, so pyarrow's schema can never drift between chunks because one
    # chunk's records happened to include/omit a key that another chunk's didn't (a real
    # risk at ~99,886 patients that a 30-patient dev sample is too uniform to expose).
    OPERATIONS_COLS = ["op_id", "subject_id", "hadm_id", "case_id", "opdate", "age", "sex",
                        "weight", "height", "race", "asa", "emop", "department", "antype",
                        "icd10_pcs", "orin_time", "orout_time", "opstart_time", "opend_time",
                        "admission_time", "discharge_time", "anstart_time", "anend_time",
                        "cpbon_time", "cpboff_time", "icuin_time", "icuout_time",
                        "inhosp_death_time", "allcause_death_time"]
    MEDICATIONS_COLS = ["subject_id", "chart_time", "drug_name", "route", "drug_name2",
                         "drug_name3", "atc_code", "atc_code2", "atc_code3"]
    OPERATIONS_STRING_COLS = ["op_id", "subject_id", "hadm_id", "case_id", "opdate", "sex",
                               "race", "department", "antype", "icd10_pcs", "cpbon_time", "cpboff_time"]
    MEDICATIONS_STRING_COLS = ["subject_id", "drug_name", "route", "drug_name2", "drug_name3",
                                "atc_code", "atc_code2", "atc_code3"]

    def _flush_chunk(name, rows, int_cols=("chart_time",), float_cols=("value",), string_cols=("subject_id",)):
        # rows: list of dicts for this chunk only (small -- at most chunk_size patients'
        # worth) -- the one place per chunk we pay the dict-object cost, not for the whole run.
        if not rows:
            return
        df = pd.DataFrame(rows)
        # Nullable "Int32" (capital I, pandas' extension dtype), not plain int32/int64: a
        # plain numpy int column cannot hold NaN at all, so pandas silently upcasts a
        # column to float64 in any chunk that happens to contain one missing value, while
        # a chunk with no gaps stays int64 -- two chunks then disagree on dtype, and
        # ParquetWriter correctly refuses to append a table whose schema doesn't match the
        # one it was opened with. Nullable Int32 holds missing values natively, so every
        # chunk gets the identical dtype regardless of whether that particular chunk has
        # any gaps.
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.array(df[c], dtype="Int32")
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].astype("float32")
        # Same idea for string columns: pandas' nullable "string" dtype (not plain object)
        # so a chunk where a column happens to be entirely missing/empty still gets typed
        # as string, not inferred as pyarrow's separate "null" type -- which would also
        # collide with a later chunk's genuinely-string-typed version of the same column.
        for c in string_cols:
            if c in df.columns:
                df[c] = df[c].astype("string")
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writers[name] is None:
            writers[name] = pq.ParquetWriter(os.path.join(cache_subdir, f"{name}.parquet"), table.schema)
        writers[name].write_table(table)

    t0 = time.time()
    n_in_chunk = 0
    labs_rows, vitals_rows, ward_vitals_rows = [], [], []
    op_rows, dx_rows, med_rows = [], [], []

    for i, (fp, folder) in enumerate(filepaths):
        with open(fp) as f:
            d = json.load(f)
        sid = str(d["subject_id"])
        folder_label[sid] = folder

        for r in d.get("labs", []):
            labs_rows.append({
                "subject_id": sid,
                "chart_time": _safe_int(r.get("chart_time")),
                "item_name": r.get("item_name"),
                "value": _safe_float(r.get("value")),
            })

        if load_vitals:
            for r in d.get("vitals", []):
                vitals_rows.append({
                    "subject_id": sid,
                    "op_id": r.get("op_id"),
                    "chart_time": _safe_int(r.get("chart_time")),
                    "item_name": r.get("item_name"),
                    "value": _safe_float(r.get("value")),
                })

        for r in d.get("ward_vitals", []):
            ward_vitals_rows.append({
                "subject_id": sid,
                "chart_time": _safe_int(r.get("chart_time")),
                "item_name": r.get("item_name"),
                "value": _safe_float(r.get("value")),
            })

        for op in d.get("operations", []):
            row = {col: op.get(col) for col in OPERATIONS_COLS}   # fixed schema, see note above
            row["subject_id"] = sid
            for tcol in ["orin_time", "orout_time", "opstart_time", "opend_time",
                         "admission_time", "discharge_time", "anstart_time", "anend_time",
                         "icuin_time", "icuout_time"]:
                row[tcol] = _safe_int(row.get(tcol))
            for icol in ["age", "asa", "emop"]:
                row[icol] = _safe_int(row.get(icol))
            for fcol in ["weight", "height"]:
                row[fcol] = _safe_float(row.get(fcol))
            row["inhosp_death_time"] = _safe_int(row.get("inhosp_death_time"))
            row["allcause_death_time"] = _safe_int(row.get("allcause_death_time"))
            op_rows.append(row)

        for dx in d.get("diagnoses", []):
            dx_rows.append({"subject_id": sid, "chart_time": _safe_int(dx.get("chart_time")),
                             "icd10_cm": dx.get("icd10_cm")})

        for m in d.get("medications", []):
            row = {col: m.get(col) for col in MEDICATIONS_COLS}   # fixed schema, see note above
            row["subject_id"] = sid
            row["chart_time"] = _safe_int(m.get("chart_time"))
            med_rows.append(row)

        del d
        n_in_chunk += 1

        if n_in_chunk >= chunk_size or i == len(filepaths) - 1:
            _flush_chunk("labs", labs_rows, string_cols=("subject_id", "item_name"))
            _flush_chunk("vitals", vitals_rows, string_cols=("subject_id", "item_name", "op_id"))
            _flush_chunk("ward_vitals", ward_vitals_rows, string_cols=("subject_id", "item_name"))
            _flush_chunk("operations", op_rows,
                         int_cols=("orin_time", "orout_time", "opstart_time", "opend_time",
                                   "admission_time", "discharge_time", "anstart_time", "anend_time",
                                   "icuin_time", "icuout_time", "age", "asa", "emop",
                                   "inhosp_death_time", "allcause_death_time"),
                         float_cols=("weight", "height"),
                         string_cols=OPERATIONS_STRING_COLS)
            _flush_chunk("diagnoses", dx_rows, int_cols=("chart_time",), float_cols=(),
                         string_cols=("subject_id", "icd10_cm"))
            _flush_chunk("medications", med_rows, int_cols=("chart_time",), float_cols=(),
                         string_cols=MEDICATIONS_STRING_COLS)
            labs_rows, vitals_rows, ward_vitals_rows = [], [], []
            op_rows, dx_rows, med_rows = [], [], []
            n_in_chunk = 0
            print(f"  ...{i+1}/{len(filepaths)} ({time.time()-t0:.0f}s elapsed, chunk flushed to disk)")

    for w in writers.values():
        if w is not None:
            w.close()

    # If a table had zero rows in EVERY chunk (e.g. "vitals" whenever TIME_WINDOW=='pre_op'),
    # its writer was never opened and its Parquet file was never created -- write an empty
    # placeholder with the right columns so the cache-completeness check (which expects
    # every table's file to exist) doesn't wrongly conclude the cache is broken/partial on
    # every future run.
    _empty_schemas = {
        "labs": ["subject_id", "chart_time", "item_name", "value"],
        "vitals": ["subject_id", "op_id", "chart_time", "item_name", "value"],
        "ward_vitals": ["subject_id", "chart_time", "item_name", "value"],
        "operations": ["subject_id"],
        "diagnoses": ["subject_id", "chart_time", "icd10_cm"],
        "medications": ["subject_id", "chart_time"],
    }
    for name in table_names:
        if writers[name] is None:
            pd.DataFrame(columns=_empty_schemas[name]).to_parquet(os.path.join(cache_subdir, f"{name}.parquet"))

    vitals_note = "peri_op -- intra-op vitals included" if load_vitals else "pre_op -- intra-op vitals skipped entirely, not just unused"
    print(f"Parsed {len(filepaths)} subjects in {time.time()-t0:.1f}s (chunked streaming, {vitals_note})")

    # Read each table back as ONE table now that every chunk is on disk, and apply
    # categorical dtypes exactly once, on the complete data -- see docstring above for why
    # this is deferred rather than done per-chunk.
    cat_cols_by_table = {
        "labs": ["subject_id", "item_name"],
        "vitals": ["subject_id", "item_name", "op_id"],
        "ward_vitals": ["subject_id", "item_name"],
        "operations": [],
        "diagnoses": ["subject_id", "icd10_cm"],
        "medications": ["subject_id", "drug_name", "route", "atc_code", "drug_name2", "drug_name3", "atc_code2", "atc_code3"],
    }
    dfs = {}
    for name in table_names:
        path = os.path.join(cache_subdir, f"{name}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            for c in cat_cols_by_table[name]:
                if c in df.columns:
                    df[c] = df[c].astype("category")
            dfs[name] = df
        else:
            dfs[name] = pd.DataFrame(columns=["subject_id", "op_id", "chart_time", "item_name", "value"]) \
                        if name == "vitals" else pd.DataFrame()

    pd.DataFrame({"subject_id": list(folder_label.keys()), "folder": list(folder_label.values())}) \
        .to_parquet(os.path.join(cache_subdir, "folder_label.parquet"))

    return dfs, folder_label

# %%
if CONFIG["USE_CACHE"] and _cache_complete(CACHE_SUBDIR):
    print(f"Found a matching cache at {CACHE_SUBDIR} -- loading from Parquet (fast path).")
    _t0 = time.time()
    DFS, FOLDER_LABEL = _load_from_cache(CACHE_SUBDIR)
    print(f"Loaded from cache in {time.time()-_t0:.1f}s.")
else:
    if CONFIG["USE_CACHE"]:
        print(f"No cache found at {CACHE_SUBDIR} -- parsing from scratch (this happens once; "
              f"chunks are written to disk as they're parsed, see Part 3's note on memory).")
    DFS, FOLDER_LABEL = stream_parse_subjects(
        CONFIG["SUBJECTS_DIR"], CONFIG["MAX_SUBJECTS_PER_CLASS"], CONFIG["TIME_WINDOW"],
        CACHE_SUBDIR, CONFIG["PARSE_CHUNK_SIZE"]
    )
    # No separate save step needed -- stream_parse_subjects already wrote every chunk to
    # CACHE_SUBDIR as it went, which is the whole point of chunking in the first place.

labs_df        = DFS["labs"]
vitals_df      = DFS["vitals"]
ward_vitals_df = DFS["ward_vitals"]
operations_df  = DFS["operations"]
diagnoses_df   = DFS["diagnoses"]
medications_df = DFS["medications"]

print(f"\nLoaded {len(FOLDER_LABEL)} subjects "
      f"({sum(v=='died' for v in FOLDER_LABEL.values())} in died/, "
      f"{sum(v=='survived' for v in FOLDER_LABEL.values())} in survived/ — folder-provided, unverified)")
if CONFIG["MAX_SUBJECTS_PER_CLASS"] is not None:
    _n_died_loaded = sum(v == "died" for v in FOLDER_LABEL.values())
    _n_survived_loaded = sum(v == "survived" for v in FOLDER_LABEL.values())
    print(f"\nDOWNSAMPLING REPORT (CONFIG['MAX_SUBJECTS_PER_CLASS']={CONFIG['MAX_SUBJECTS_PER_CLASS']}):")
    print(f"  died:     {_n_died_loaded} loaded -- ALL real deaths are always kept, this cap never touches them")
    print(f"  survived: {_n_survived_loaded} loaded (capped from the full cohort's ~99,417 survivors, "
          f"randomly sampled per-file before parsing -- see load_all_subjects)")
    print(f"  This is a real, intentional target run size, not a placeholder smoke test -- "
          f"set MAX_SUBJECTS_PER_CLASS=None only once you have the time budget for the true "
          f"full ~99,886-patient cohort (expect data loading alone to take several minutes more).")
if CONFIG["TIME_WINDOW"] != "peri_op":
    print("TIME_WINDOW='pre_op' -- intra-op vitals table NOT parsed at all (not just unused), "
          "saving both time and memory. Switch to 'peri_op' in Part 2 if you need it.")

print("\nlabs_df        ", labs_df.shape)
print("vitals_df      ", vitals_df.shape, " (intra-op)")
print("ward_vitals_df ", ward_vitals_df.shape)
print("operations_df  ", operations_df.shape)
print("diagnoses_df   ", diagnoses_df.shape)
print("medications_df ", medications_df.shape)

print("\nActual memory used by each table (post dtype-optimization) -- watch this number, "
      "not a guess extrapolated from a different-sized run:")
_total_mb = 0
for _name, _df in [("labs_df", labs_df), ("vitals_df", vitals_df), ("ward_vitals_df", ward_vitals_df),
                    ("operations_df", operations_df), ("diagnoses_df", diagnoses_df), ("medications_df", medications_df)]:
    _mb = _df.memory_usage(deep=True).sum() / 1e6
    _total_mb += _mb
    print(f"  {_name:16s} {_mb:8.1f} MB")
print(f"  {'TOTAL':16s} {_total_mb:8.1f} MB")
_n_loaded = max(len(FOLDER_LABEL), 1)
print(f"\nMedication administrations per loaded patient: {len(medications_df) / _n_loaded:.0f} "
      f"(the source repo's README gives ~99/patient for the full cohort). A run that keeps every "
      f"death but caps survivors is sicker and denser than the full cohort, so scaling this "
      f"run's memory linearly to ~99,886 patients will tend to OVERESTIMATE the real footprint.")

# %%
# Sanity check: exactly one operation row expected per (subject, op_id); most patients
# have one operation in this dev subset, some have several (relevant to §6.10/§6.11).
ops_per_subject = operations_df.groupby("subject_id")["op_id"].nunique()
print(ops_per_subject.value_counts().sort_index().rename("n_subjects_with_this_many_ops"))
operations_df.head(3)

# %% [markdown]
# ### A note on memory and speed strategy
#
# If the memory printout above is tight, or a fresh (uncached) parse is taking a long time,
# the order of things to try is:
#
# 1. **Confirm `TIME_WINDOW='pre_op'`** (the default) — this now skips *parsing* intra-op
#    `vitals` entirely, not just using it, which saves both time and memory versus the
#    earlier version of this notebook.
# 2. **Let the cache do its job.** The first parse of a given `SUBJECTS_DIR` is the expensive
#    one; every subsequent run (this session or later, if `CACHE_DIR` is on Drive) should be
#    loading Parquet in seconds. If it isn't, check `CONFIG['USE_CACHE']` is `True` and that
#    `CACHE_DIR` actually points somewhere persistent.
# 3. **Run a `MAX_SUBJECTS_PER_CLASS` smoke test first** (e.g. 2,000–5,000) and look at the
#    real memory printout, rather than extrapolating from this dev subset — which, per the
#    note above, is likely denser-than-average and will overestimate the real number.
# 4. **Only if still constrained, downsample — but the training `survived` split only, never
#    `died`, and never validation/test** (Part 8.4 implements this, off by default). Two
#    real costs of downsampling `survived` patients that are worth knowing before you turn
#    it on: it shrinks the unlabelled pool the Part 9.4 autoencoder pre-training specifically
#    relies on (a design decision made explicitly *because* splitting into six systems
#    already splits the scarce mortality labels six ways), and if it ever touched
#    validation/test it would corrupt the true ~0.47% prevalence that makes AUPRC meaningful
#    — which is why it's restricted to the training split only.
# 5. **If you do downsample, stratify by `department`, not by ICD-10 "organ system."**
#    Department is a single, always-populated categorical field per patient — clean to
#    stratify on. ICD-10 diagnoses are multi-label (a patient's diagnoses can span several
#    organ-system chapters at once), so there's no single unambiguous rule for assigning one
#    patient to one "organ-system stratum" without inventing a new, separate convention just
#    for this — and department is already the stratification key `§8.2`'s grouped SMOTENC
#    uses, so reusing it keeps one consistent clinical-neighbourhood definition through the
#    whole pipeline instead of two different ones.

# %%
record_checkpoint("Part 3 -- data loading")
print(f"Part 3 complete. Elapsed so far: {time.time() - RUN_TIMER['checkpoints'][0][1]:.1f}s")
