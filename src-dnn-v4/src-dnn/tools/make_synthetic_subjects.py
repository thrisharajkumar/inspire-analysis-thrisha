"""
Generate a FAKE INSPIRE-format cohort (died/ and survived/ folders of per-patient JSON) for
dry-running the whole pipeline end to end without real data -- e.g. after editing a stage,
before spending Kaggle GPU time.

    python tools/make_synthetic_subjects.py --out /tmp/fake_subjects --n 800 --death-rate 0.1

It reproduces the SCHEMA the parser expects (item names, time columns, multi-operation
patients, a prior CTS operation inside 6 months, sparse pre-op SpO2, zero pre-op urine
output, deaths just outside 30 days so the label recompute matters) and plants a modest,
clinically-shaped signal so AUROC/AUPRC are not trivially 0.5. Numbers produced from this
data mean nothing clinically -- it exists to catch crashes and wiring bugs.
"""
import argparse, json, os
import numpy as np

LAB_NORMALS = {  # item: (mean_survived, sd, shift_if_died)
    "bun": (15, 5, 12), "calcium": (9.2, 0.4, -0.5), "chloride": (103, 3, 1), "creatinine": (0.9, 0.25, 0.9),
    "ica": (1.15, 0.05, -0.08), "phosphorus": (3.5, 0.6, 0.8), "potassium": (4.1, 0.4, 0.4), "sodium": (139, 3, -3),
    "ck": (100, 50, 150), "ckmb": (3, 1.5, 5), "troponin_i": (0.02, 0.02, 0.3),
    "be": (0, 2, -4), "hco3": (24, 2, -4), "paco2": (40, 4, 3), "pao2": (95, 12, -20), "ph": (7.40, 0.03, -0.07), "sao2": (97, 1.5, -4),
    "albumin": (4.0, 0.4, -0.9), "alp": (80, 25, 40), "alt": (25, 12, 30), "ast": (25, 10, 35), "glucose": (110, 25, 40),
    "hba1c": (5.8, 0.6, 0.6), "lacate": (1.2, 0.4, 1.6), "total_bilirubin": (0.7, 0.3, 1.0), "total_protein": (7.0, 0.5, -0.7),
    "aptt": (32, 4, 8), "crp": (0.5, 0.6, 5), "fibrinogen": (300, 60, 60), "hb": (13, 1.5, -2.5), "hct": (39, 4, -7),
    "lymphocyte": (28, 8, -12), "platelet": (250, 60, -80), "ptinr": (1.0, 0.08, 0.35), "seg": (60, 8, 15), "wbc": (7, 2, 5),
}
VITAL_NORMALS = {  # item: (mean_survived, sd, shift_if_died, coverage)
    "hr": (74, 10, 22, 0.97), "nibp_sbp": (128, 15, -22, 0.97), "nibp_dbp": (75, 9, -10, 0.97), "nibp_mbp": (92, 10, -15, 0.97),
    "rr": (17, 2, 5, 0.93), "spo2": (97.5, 1.2, -4, 0.40), "bt": (36.7, 0.3, 0.8, 0.95),
    "fio2": (21, 0.5, 15, 0.15), "gcs_e": (4, 0, -1, 0.10), "gcs_m": (6, 0, -1, 0.10), "gcs_v": (5, 0, -1, 0.10),
}
DEPTS = ["GS", "OS", "CTS", "NS", "UR", "OL", "OG", "PS"]
DEPT_RISK = {"GS": 1.0, "OS": 0.6, "CTS": 2.5, "NS": 1.6, "UR": 0.6, "OL": 0.8, "OG": 0.3, "PS": 0.4}
ICD_POOL = {"GS": ["K80", "K35", "K56", "C18"], "OS": ["M16", "M17", "S72", "M48"], "CTS": ["I25", "I35", "I21", "I50"],
            "NS": ["G93", "I61", "C71"], "UR": ["N20", "C67", "N40"], "OL": ["J34", "H66"], "OG": ["O82", "N80"], "PS": ["L90", "T30"]}
PCS_POOL = {"GS": ["0DTJ4ZZ", "0DB60ZZ", "0FT44ZZ"], "OS": ["0SR90J9", "0QS604Z", "0SG00A0"], "CTS": ["021109W", "02RF0JZ"],
            "NS": ["00B00ZZ", "0RG10A0"], "UR": ["0TT10ZZ", "0VT08ZZ"], "OL": ["09BM8ZZ"], "OG": ["0UT90ZZ"], "PS": ["0HBJXZZ"]}
ANTYPES = ["General", "General", "General", "Spinal", "MAC"]
RISK_CODES = ["A41", "R57", "J80", "N17", "I46", "D65", "E87", "R40"]
DRUGS = [("cefazolin", "J01DB04"), ("acetaminophen", "N02BE01"), ("heparin", "B01AB01"), ("insulin", "A10AB01"),
         ("norepinephrine", "C01CA03"), ("fentanyl", "N01AH01"), ("furosemide", "C03CA01"), ("pantoprazole", "A02BC02"),
         ("ibuprofen", "M01AE01"), ("salbutamol", "R03AC02")]
MIN_PER_DAY = 1440


def make_patient(sid, died, rng):
    dept = rng.choice(DEPTS, p=np.array([.25, .2, .1, .1, .1, .1, .1, .05]))
    sev = rng.normal(1.0 if died else 0.0, 0.6) * (1.0 if died else 0.5)     # latent severity
    asa = int(np.clip(round(2 + sev + rng.normal(0, 0.6)), 1, 5))
    age = int(np.clip(rng.normal(68 if died else 57, 13), 18, 95))
    orin = int(rng.integers(30, 400) * MIN_PER_DAY + rng.integers(0, MIN_PER_DAY))
    admission = orin - int(rng.integers(1, 6) * MIN_PER_DAY)
    ops = []
    if rng.random() < 0.12:   # earlier operation; for CTS sometimes inside the 6-month recovery window
        gap_days = int(rng.integers(20, 170) if (dept == "CTS" and rng.random() < 0.6) else rng.integers(200, 700))
        p_orin = orin - gap_days * MIN_PER_DAY
        ops.append(dict(op_id=f"{sid}_0", orin=p_orin, dept=dept if rng.random() < 0.7 else rng.choice(DEPTS)))
    ops.append(dict(op_id=f"{sid}_1", orin=orin, dept=dept))
    last_orout = orin + int(rng.integers(60, 400))
    death = None
    if died:
        death = last_orout + int(rng.integers(1, 28) * MIN_PER_DAY)
        if rng.random() < 0.08:   # dies after day 30: in died/ folder, but NOT a 30-day death
            death = last_orout + int(rng.integers(35, 80) * MIN_PER_DAY)
    operations = []
    for o in ops:
        orout = o["orin"] + int(rng.integers(60, 400)) if o["orin"] != orin else last_orout
        operations.append({
            "op_id": o["op_id"], "subject_id": sid, "hadm_id": f"h{sid}", "case_id": f"c{o['op_id']}",
            "opdate": o["orin"] // MIN_PER_DAY, "age": age, "sex": rng.choice(["M", "F"]),
            "weight": round(float(rng.normal(65, 12)), 1), "height": round(float(rng.normal(165, 9)), 1),
            "race": "Asian", "asa": asa, "emop": int(rng.random() < (0.35 if died else 0.1)), "department": o["dept"],
            "antype": str(rng.choice(ANTYPES)), "icd10_pcs": str(rng.choice(PCS_POOL[o["dept"]])), "orin_time": o["orin"], "orout_time": orout,
            "opstart_time": o["orin"] + 15, "opend_time": orout - 10, "admission_time": admission,
            "discharge_time": orout + 7 * MIN_PER_DAY, "anstart_time": o["orin"] + 5, "anend_time": orout - 5,
            "cpbon_time": None, "cpboff_time": None, "icuin_time": None, "icuout_time": None,
            "inhosp_death_time": death, "allcause_death_time": death,
        })
    lo = orin - 5 * MIN_PER_DAY
    labs, ward = [], []
    for item, (mu, sd, shift) in LAB_NORMALS.items():
        if rng.random() < (0.55 if item in ("creatinine", "hb", "wbc", "sodium", "potassium", "glucose", "platelet") else 0.25):
            for _ in range(int(rng.integers(1, 4))):
                t = int(rng.integers(lo, orin))
                labs.append({"chart_time": t, "item_name": item, "value": round(float(rng.normal(mu + shift * max(sev, 0), sd)), 3)})
    n_sets = int(rng.integers(4, 14))
    for k in range(n_sets):
        t = int(lo + (orin - lo) * (k + rng.random()) / n_sets)
        for item, (mu, sd, shift, cov) in VITAL_NORMALS.items():
            if rng.random() < cov:
                v = rng.normal(mu + shift * max(sev, 0) * (k + 1) / n_sets, sd)
                if item.startswith("gcs"):
                    v = round(v)
                ward.append({"chart_time": t + int(rng.integers(0, 5)), "item_name": item, "value": round(float(v), 1)})
    if dept == "CTS" and rng.random() < 0.05:
        ward.append({"chart_time": orin - 60, "item_name": "iabp", "value": 1})
    diagnoses = [{"chart_time": int(rng.integers(lo, orin)), "icd10_cm": str(rng.choice(ICD_POOL[dept]))}]
    if rng.random() < (0.5 if died else 0.08):
        diagnoses.append({"chart_time": int(rng.integers(lo, orin)), "icd10_cm": str(rng.choice(RISK_CODES))})
    if age >= 75 and rng.random() < 0.5:
        diagnoses.append({"chart_time": int(rng.integers(lo, orin)), "icd10_cm": str(rng.choice(["F05", "R26", "N39", "S72"]))})
    meds = []
    for _ in range(int(rng.integers(2, 12))):
        name, atc = DRUGS[int(rng.choice([0, 1, 2, 3, 8, 9] + ([4, 5, 6, 7] if died else [])))]
        meds.append({"chart_time": int(rng.integers(lo, orin)), "drug_name": name, "route": "IV",
                     "drug_name2": None, "drug_name3": None, "atc_code": atc, "atc_code2": None, "atc_code3": None})
    return {"subject_id": sid, "operations": operations, "labs": labs, "vitals": [], "ward_vitals": ward,
            "diagnoses": diagnoses, "medications": meds}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--death-rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for folder in ("died", "survived"):
        os.makedirs(os.path.join(a.out, folder), exist_ok=True)
    n_died = int(a.n * a.death_rate)
    for i in range(a.n):
        died = i < n_died
        sid = str(100000 + i)
        with open(os.path.join(a.out, "died" if died else "survived", f"{sid}.json"), "w") as f:
            json.dump(make_patient(sid, died, rng), f)
    print(f"Wrote {a.n} fake patients ({n_died} in died/) to {a.out}")
