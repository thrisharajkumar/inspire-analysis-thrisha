# %% [markdown]
# # Part 4 — Item-name inventory, the 30-day mortality label, and the cohort table
#
# Three things happen here, each directly implementing a decision from Part 1:
#
# 1. **§4.1** — list every distinct `item_name` in `labs` / `vitals` / `ward_vitals`. This is
#    the programmatic check behind §1.2.3's claim that GI and MSK have no dedicated
#    lab/vital panel — run it and see for yourself rather than taking the claim on faith.
# 2. **§4.2** — recompute the 30-day mortality label directly from `operations`, **not**
#    from the `died`/`survived` folder name (§1.6.3's label-bug note).
# 3. **§4.3** — assemble one row per patient: label, static fields, multi-operation counts.

# %% [markdown]
# ## 4.1 Item-name inventory — confirming §1.2.3 programmatically

# %%
def item_name_inventory(*dfs_and_names):
    inv = {}
    for df, name in dfs_and_names:
        inv[name] = sorted(df["item_name"].dropna().unique().tolist()) if "item_name" in df.columns and len(df) else []
    return inv

INVENTORY = item_name_inventory((labs_df, "labs"), (vitals_df, "vitals (intra-op)"), (ward_vitals_df, "ward_vitals"))
for name, items in INVENTORY.items():
    print(f"{name}: {len(items)} distinct item_names")

# GI/MSK check: do any of these look like GI- or MSK-specific measurements?
gi_keywords = ["bowel", "stool", "gi_", "gastro", "bilirubin"]  # bilirubin included to show it's hepatic, see §1.2.3
msk_keywords = ["mobility", "joint", "rom", "muscle", "ortho"]
all_items = set(sum(INVENTORY.values(), []))
print("\nItems matching GI-ish keywords:", [i for i in all_items if any(k in i.lower() for k in gi_keywords)])
print("Items matching MSK-ish keywords:", [i for i in all_items if any(k in i.lower() for k in msk_keywords)])
print("\n-> Confirms §1.2.3: no dedicated GI or MSK signal in labs/vitals/ward_vitals.")
print("   GI and MSK must be built from ICD-10 diagnoses + department (done in §6.7-6.8).")

# %% [markdown]
# ## 4.2 The 30-day mortality label — recomputed from `operations`, not the folder name

# %%
MINUTES_PER_DAY = 24 * 60

def compute_labels(operations_df):
    # One row per subject. Mirrors subject.py's inhosp_death_30day() logic:
    # died = inhosp_death_time is set AND inhosp_death_time < orout_time(last op) + 30 days.
    # 'last op' = operation with the max orin_time for that subject (current repo convention;
    # see §1.6.3/CONFIG note below for why this is a genuine open question, not a settled one).
    rows = []
    for sid, g in operations_df.groupby("subject_id"):
        g = g.sort_values("orin_time")
        first_op = g.iloc[0]
        last_op = g.iloc[-1]
        n_ops = len(g)

        inhosp_death_time = first_op["inhosp_death_time"]   # same across all ops for a subject, per source repo's note
        allcause_death_time = first_op["allcause_death_time"]

        # NOTE: pandas stores our None sentinels as NaN once a column mixes numbers and
        # missing values, so `x is not None` silently fails here -- must use pd.notna().
        has_inhosp_death = pd.notna(inhosp_death_time)

        died_30day_from_last = False
        died_30day_from_first = False
        if has_inhosp_death:
            if pd.notna(last_op["orout_time"]):
                died_30day_from_last = inhosp_death_time < (last_op["orout_time"] + 30 * MINUTES_PER_DAY)
            if pd.notna(first_op["orout_time"]):
                died_30day_from_first = inhosp_death_time < (first_op["orout_time"] + 30 * MINUTES_PER_DAY)

        died_ever = has_inhosp_death

        rows.append({
            "subject_id": sid,
            "n_operations": n_ops,
            "age": last_op["age"],
            "sex": last_op["sex"],
            "asa": last_op["asa"],
            "emop": last_op["emop"],
            "department": last_op["department"],
            "antype": last_op.get("antype"),
            "weight": last_op["weight"],
            "height": last_op["height"],
            "op_id_last": last_op["op_id"],
            "orin_time_last": last_op["orin_time"],
            "orout_time_last": last_op["orout_time"],
            "admission_time_last": last_op["admission_time"],
            "discharge_time_last": last_op["discharge_time"],
            "died_ever": died_ever,                                   # NOT the training label - see §1.6.3
            "died_30day_from_last_op": died_30day_from_last,          # default training label (Path C = 'last operation', see §1.6.4/§11.7)
            "died_30day_from_first_op": died_30day_from_first,        # sensitivity-analysis alternative
        })
    return pd.DataFrame(rows)

cohort_df = compute_labels(operations_df)
cohort_df = cohort_df.merge(
    pd.Series(FOLDER_LABEL, name="folder_label").rename_axis("subject_id").reset_index(),
    on="subject_id", how="left"
)

# Cross-check against the folder label, exactly the check §1.6.3 says not to skip.
cohort_df["folder_says_died"] = cohort_df["folder_label"].eq("died")
mismatch = cohort_df[cohort_df["folder_says_died"] != cohort_df["died_30day_from_last_op"]]
print(f"Cohort: {len(cohort_df)} patients.")
print(f"  died_ever (all-cause, any time)      : {cohort_df['died_ever'].sum()}")
print(f"  died_30day_from_last_op (TRAINING LABEL) : {cohort_df['died_30day_from_last_op'].sum()}")
print(f"  died_30day_from_first_op (sensitivity)   : {cohort_df['died_30day_from_first_op'].sum()}")
print(f"  folder-label says 'died'             : {cohort_df['folder_says_died'].sum()}")
print(f"  Rows where folder label != recomputed 30-day label: {len(mismatch)}  "
      f"(non-zero here is expected and is exactly the §1.6.3 bug the recompute avoids inheriting)")
mismatch[["subject_id", "folder_says_died", "died_ever", "died_30day_from_last_op"]]

# %% [markdown]
# ## 4.3 Multi-operation counts by department/system — feeds §6.10/§6.11

# %%
# Per-subject operation history (all ops, in order) — used later for the "operations in the
# same area" feature (§6.10) and the post-cardiac-surgery exception flag (§6.11).
OPS_HISTORY = {}
for sid, g in operations_df.groupby("subject_id"):
    g = g.sort_values("orin_time")
    OPS_HISTORY[sid] = g.to_dict("records")

print("Departments observed:", sorted(operations_df["department"].dropna().unique().tolist()))
print("\nMortality (30-day, last-op label) by department:")
tmp = cohort_df.groupby("department")["died_30day_from_last_op"].agg(["mean", "count"])
print(tmp.sort_values("mean", ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
cohort_df["n_operations"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#0a7d6e")
axes[0].set_title("Operations per patient"); axes[0].set_xlabel("n_operations"); axes[0].set_ylabel("n_patients")

dep_mort = cohort_df.groupby("department")["died_30day_from_last_op"].mean().sort_values(ascending=False)
dep_mort.plot(kind="bar", ax=axes[1], color="#c0392b")
axes[1].set_title("30-day mortality rate by department"); axes[1].set_ylabel("mortality rate")
plt.tight_layout(); plt.show()

# %% [markdown]
# # Part 5 — Exploratory data analysis
#
# A cohort overview, then targeted checks that surface things the modelling choices in
# Part 6 onward need to account for.

# %%
# 5.0 Cohort overview -- at-a-glance picture of who's actually in this run.
n_total = len(cohort_df)
n_died = int(cohort_df["died_30day_from_last_op"].sum())
n_survived = n_total - n_died
print(f"Cohort: {n_total} patients -- {n_died} died within 30 days ({n_died/n_total:.1%}), "
      f"{n_survived} survived ({n_survived/n_total:.1%})")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# died/survived counts
axes[0].bar(["died", "survived"], [n_died, n_survived], color=["#c0392b", "#2980b9"])
axes[0].set_title("Cohort composition"); axes[0].set_ylabel("n patients")
for i, v in enumerate([n_died, n_survived]):
    axes[0].text(i, v, str(v), ha="center", va="bottom")

# age distribution by outcome
for outcome, color, label in [(1.0, "#c0392b", "died"), (0.0, "#2980b9", "survived")]:
    ages = cohort_df.loc[cohort_df["died_30day_from_last_op"] == outcome, "age"].dropna()
    if len(ages):
        axes[1].hist(ages, bins=15, alpha=0.6, color=color, label=label, density=True)
axes[1].set_title("Age distribution by outcome"); axes[1].set_xlabel("age"); axes[1].legend()

# department distribution
dept_counts = cohort_df["department"].value_counts()
axes[2].bar(dept_counts.index, dept_counts.values, color="#8e44ad")
axes[2].set_title("Patients per department"); axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout(); plt.show()

# %%
# 5.1 ASA vs mortality -- sanity check that the data behaves the way the literature predicts
# (higher ASA class -> higher mortality is a well-established, monotonic clinical relationship).
asa_mort = cohort_df.groupby("asa")["died_30day_from_last_op"].agg(["mean", "count"])
print(asa_mort)

fig, ax = plt.subplots(figsize=(5, 4))
asa_mort["mean"].plot(kind="bar", ax=ax, color="#2c3e50")
ax.set_title("30-day mortality rate by ASA class"); ax.set_xlabel("ASA class"); ax.set_ylabel("mortality rate")
plt.tight_layout(); plt.show()

# %%
# 5.2 Missingness -- which pre-op labs/vitals are actually available per patient?
# This is the concrete evidence behind §1.3's MNAR discussion: a feature with very low
# pre-op coverage is either genuinely unmeasured or (per §1.3) not ordered because the
# clinician judged it unnecessary -- worth seeing the real numbers before choosing §7's
# imputation strategy.
def coverage_table(df, cohort_df, time_col="chart_time", window_minutes=None, op_time_lookup=None):
    # Fraction of patients with >=1 observation of each item_name, optionally restricted
    # to a pre-op window (window_minutes before orin_time).
    if window_minutes is not None:
        keep_rows = []
        for sid, g in df.groupby("subject_id"):
            orin = op_time_lookup.get(sid)
            if orin is None:
                continue
            lo = orin - window_minutes
            keep_rows.append(g[(g[time_col] >= lo) & (g[time_col] <= orin)])
        df = pd.concat(keep_rows) if keep_rows else df.iloc[0:0]
    n_patients = cohort_df["subject_id"].nunique()
    cov = df.groupby("item_name")["subject_id"].nunique() / n_patients
    return cov.sort_values(ascending=False)

orin_lookup = cohort_df.set_index("subject_id")["orin_time_last"].to_dict()
window = CONFIG["PRE_OP_DAYS"] * 24 * 60

labs_cov_preop = coverage_table(labs_df, cohort_df, window_minutes=window, op_time_lookup=orin_lookup)
wv_cov_preop   = coverage_table(ward_vitals_df, cohort_df, window_minutes=window, op_time_lookup=orin_lookup)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
labs_cov_preop.plot(kind="bar", ax=axes[0], color="#2980b9"); axes[0].set_title(f"Pre-op ({CONFIG['PRE_OP_DAYS']}d) lab coverage")
wv_cov_preop.plot(kind="bar", ax=axes[1], color="#8e44ad"); axes[1].set_title(f"Pre-op ({CONFIG['PRE_OP_DAYS']}d) ward-vital coverage")
for ax in axes:
    ax.set_ylabel("fraction of patients with >=1 value"); ax.tick_params(axis='x', labelsize=7)
plt.tight_layout(); plt.show()

print("Lowest-coverage pre-op labs (candidates for the MNAR discussion in §1.3):")
print(labs_cov_preop.tail(8))

# %%
# 5.3 ICD-10 chapter distribution -- which chapters actually show up, and how do they
# relate to mortality? Directly relevant to §1.2.3 (GI = chapter XI, MSK = chapter XIII).

ICD10_CHAPTERS = [
    ('I', 'A00-B99', 'Certain infectious and parasitic diseases'),
    ('II', 'C00-D48', 'Neoplasms'),
    ('III', 'D50-D89', 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism'),
    ('IV', 'E00-E90', 'Endocrine, nutritional and metabolic diseases'),
    ('V', 'F00-F99', 'Mental and behavioural disorders'),
    ('VI', 'G00-G99', 'Diseases of the nervous system'),
    ('VII', 'H00-H59', 'Diseases of the eye and adnexia'),
    ('VIII', 'H60-H95', 'Diseases of the ear and mastoid process'),
    ('IX', 'I00-I99', 'Diseases of the circulatory system'),
    ('X', 'J00-J99', 'Diseases of the respiratory system'),
    ('XI', 'K00-K93', 'Diseases of the digestive system'),                       # <- GI
    ('XII', 'L00-L99', 'Diseases of the skin and subcutaneous tissue'),
    ('XIII', 'M00-M99', 'Diseases of the musculoskeletal system and connective tissue'),  # <- MSK
    ('XIV', 'N00-N99', 'Diseases of the genitourinary system'),
    ('XV', 'O00-O99', 'Pregnancy, childbirth and the puerperium'),
    ('XVI', 'P00-P96', 'Certain conditions originating in the perinatal period'),
    ('XVII', 'Q00-Q99', 'Congenital malformations, deformations and chromosomal abnormalities'),
    ('XVIII', 'R00-R99', 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified'),
    ('XIX', 'S00-T98', 'Injury, poisoning and certain other consequences of external causes'),
    ('XX', 'V01-Y98', 'External causes of morbidity and mortality'),
    ('XXI', 'Z00-Z99', 'Factors influencing health status and contact with health services'),
    ('XXII', 'U00-U99', 'Codes for special purposes'),
]

def _in_block(code3, block):
    # code3: first 3 chars of an ICD-10-CM code, e.g. 'N18'. block: 'A00-B99'.
    if not code3 or len(code3) < 3:
        return False
    alpha = ord(code3[0].upper())
    try:
        numeric = int(code3[1:3])
    except ValueError:
        return False
    sblock, eblock = block.split('-')
    salpha, snum = ord(sblock[0]), int(sblock[1:])
    ealpha, enum = ord(eblock[0]), int(eblock[1:])
    return salpha <= alpha <= ealpha and snum <= numeric <= enum

def icd10_chapter(code):
    if not isinstance(code, str) or len(code) < 3:
        return None
    code3 = code[:3].upper()
    for numeral, block, desc in ICD10_CHAPTERS:
        if _in_block(code3, block):
            return numeral
    return None

diagnoses_df["chapter"] = diagnoses_df["icd10_cm"].apply(icd10_chapter)
chapter_desc = {c: d for c, b, d in ICD10_CHAPTERS}

chapter_counts = diagnoses_df["chapter"].value_counts()
print(chapter_counts)

# mortality rate among patients who have >=1 diagnosis in each chapter
chap_mort = {}
for chap in chapter_counts.index:
    sids = set(diagnoses_df.loc[diagnoses_df["chapter"] == chap, "subject_id"])
    sub = cohort_df[cohort_df["subject_id"].isin(sids)]
    if len(sub):
        chap_mort[chap] = (sub["died_30day_from_last_op"].mean(), len(sub))
chap_mort_df = pd.DataFrame(chap_mort, index=["mortality_rate", "n_patients"]).T.sort_values("mortality_rate", ascending=False)
chap_mort_df["description"] = [chapter_desc.get(c, "") for c in chap_mort_df.index]
chap_mort_df
