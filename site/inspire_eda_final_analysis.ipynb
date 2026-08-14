"""
audit_features.py

Runs across ALL patients (not just one) and checks their actual recorded
data against the real INSPIRE schema in parameters.csv. Answers:

  1. Of the features that exist in the schema, how many does EACH patient
     actually have data for -- both anywhere in their record, and
     specifically inside the 5-day pre-op window the model uses?
  2. Which specific features are missing for each patient?
  3. Across ALL patients, what % have each feature at all, and what %
     have it inside the pre-op window?

USAGE (Colab):
    !python audit_features.py \\
        /content/inspire_subjects_small/inspire_subjects_small \\
        parameters.csv

"""

import csv
import json
import os
import sys
from collections import defaultdict

DAYS_BEFORE_OPERATION = 5
MINUTES_BEFORE_OPERATION = DAYS_BEFORE_OPERATION * 24 * 60  # 7200


# --------------------------------------------------------------------
# Load the ground-truth schema (what SHOULD exist), if available
# --------------------------------------------------------------------
def load_schema(parameters_csv_path):
    """Returns {'labs': [...], 'ward_vitals': [...], 'vitals': [...]}"""
    schema = defaultdict(list)
    if not parameters_csv_path or not os.path.isfile(parameters_csv_path):
        return None
    with open(parameters_csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            schema[row["Table"]].append(row["Label"])
    return dict(schema)


# --------------------------------------------------------------------
# Find patient JSON files (same convention as explore_features.py)
# --------------------------------------------------------------------
def find_json_files(path):
    files = []
    if os.path.isfile(path):
        files.append((path, "unknown"))
        return files

    survived_dir = os.path.join(path, "survived")
    died_dir = os.path.join(path, "died")

    if os.path.isdir(survived_dir) or os.path.isdir(died_dir):
        for label, folder in [("survived", survived_dir), ("died", died_dir)]:
            if os.path.isdir(folder):
                for fname in sorted(os.listdir(folder)):
                    if fname.endswith(".json"):
                        files.append((os.path.join(folder, fname), label))
    else:
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json"):
                files.append((os.path.join(path, fname), "unknown"))
    return files


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------
# Per-patient audit
# --------------------------------------------------------------------
def audit_patient(data, schema):
    """
    Returns a dict describing, for this one patient:
      - subject_id, label, department
      - for labs+ward_vitals (the pre-op-usable tables):
          present_anywhere: set of feature names found anywhere in record
          present_in_window: set of feature names found in the 5-day pre-op window
          record_counts_in_window: {feature: count}
    """
    subject_id = data.get("subject_id", "UNKNOWN")

    operations = data.get("operations") or []
    department = operations[0].get("department", "UNKNOWN") if operations else "UNKNOWN"

    orin_time = safe_float(operations[0].get("orin_time")) if operations else None
    window_start = orin_time - MINUTES_BEFORE_OPERATION if orin_time is not None else None
    window_end = orin_time - 1 if orin_time is not None else None

    present_anywhere = defaultdict(set)     # table -> set of feature names
    present_in_window = defaultdict(set)    # table -> set of feature names
    counts_in_window = defaultdict(int)     # (table, feature) -> count

    for table in ["labs", "ward_vitals", "vitals"]:
        for record in data.get(table, []):
            name = record.get("item_name")
            if name is None:
                continue
            present_anywhere[table].add(name)

            if window_start is not None:
                t = safe_float(record.get("chart_time"))
                if t is not None and window_start <= t <= window_end:
                    present_in_window[table].add(name)
                    counts_in_window[(table, name)] += 1

    return {
        "subject_id": subject_id,
        "department": department,
        "orin_time": orin_time,
        "present_anywhere": present_anywhere,
        "present_in_window": present_in_window,
        "counts_in_window": counts_in_window,
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(data_path, parameters_csv_path=None):
    schema = load_schema(parameters_csv_path)
    files = find_json_files(data_path)
    if not files:
        print(f"No JSON files found at '{data_path}'. Check the path.")
        return

    print(f"Found {len(files)} patient file(s)\n")

    preop_tables = ["labs", "ward_vitals"]
    if schema:
        preop_schema_features = {
            (table, name) for table in preop_tables for name in schema.get(table, [])
        }
        print(f"Schema says {len(preop_schema_features)} pre-op-usable feature types "
              f"should exist ({len(schema.get('labs', []))} labs + "
              f"{len(schema.get('ward_vitals', []))} ward_vitals)\n")
    else:
        preop_schema_features = None
        print("No parameters.csv provided -- will report what's found, "
              "but can't tell you what's MISSING relative to the full schema.\n")

    per_patient_results = []
    label_counts = defaultdict(int)
    dept_counts = defaultdict(int)

    for filepath, label in files:
        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Skipping {filepath}: could not read ({e})")
            continue
        label_counts[label] += 1
        result = audit_patient(data, schema)
        result["label"] = label
        dept_counts[result["department"]] += 1
        per_patient_results.append(result)

    n_patients = len(per_patient_results)

    # ---------------- Per-patient completeness ----------------
    print("=" * 90)
    print("PER-PATIENT COMPLETENESS (features present in the 5-day pre-op window)")
    print("=" * 90)
    completeness_scores = []
    for r in per_patient_results:
        found_in_window = set()
        for table in preop_tables:
            for name in r["present_in_window"].get(table, set()):
                found_in_window.add((table, name))

        if preop_schema_features:
            n_found = len(found_in_window)
            n_total = len(preop_schema_features)
            missing = sorted(name for (table, name) in (preop_schema_features - found_in_window))
            completeness_scores.append(n_found)
            print(f"  subject {r['subject_id']:<12} label={r['label']:<9} dept={r['department']:<5} "
                  f"{n_found}/{n_total} schema features present in pre-op window")
            if missing:
                print(f"      missing: {', '.join(missing)}")
        else:
            n_found = len(found_in_window)
            completeness_scores.append(n_found)
            print(f"  subject {r['subject_id']:<12} label={r['label']:<9} dept={r['department']:<5} "
                  f"{n_found} distinct feature types present in pre-op window")

    if completeness_scores:
        avg = sum(completeness_scores) / len(completeness_scores)
        print()
        print(f"Completeness across {n_patients} patients: "
              f"mean={avg:.1f}, min={min(completeness_scores)}, max={max(completeness_scores)}")

    # ---------------- Per-feature coverage across all patients ----------------
    print()
    print("=" * 90)
    print("PER-FEATURE COVERAGE ACROSS ALL PATIENTS")
    print("=" * 90)
    print(f"{'FEATURE':<18}{'TABLE':<14}{'IN WINDOW (patients)':<24}{'ANYWHERE (patients)':<22}")
    print("-" * 90)

    all_features = set()
    for r in per_patient_results:
        for table in preop_tables:
            for name in r["present_anywhere"].get(table, set()):
                all_features.add((table, name))
    if preop_schema_features:
        all_features |= preop_schema_features

    rows = []
    for table, name in sorted(all_features, key=lambda kv: kv[1]):
        n_anywhere = sum(1 for r in per_patient_results if name in r["present_anywhere"].get(table, set()))
        n_in_window = sum(1 for r in per_patient_results if name in r["present_in_window"].get(table, set()))
        rows.append((table, name, n_in_window, n_anywhere))

    rows.sort(key=lambda row: -row[2])  # most-covered-in-window first
    for table, name, n_in_window, n_anywhere in rows:
        pct_window = 100 * n_in_window / n_patients
        pct_anywhere = 100 * n_anywhere / n_patients
        print(f"{name:<18}{table:<14}{f'{n_in_window}/{n_patients} ({pct_window:.0f}%)':<24}"
              f"{f'{n_anywhere}/{n_patients} ({pct_anywhere:.0f}%)':<22}")

    # ---------------- Features missing for EVERYONE ----------------
    if preop_schema_features:
        never_found = sorted(
            name for (table, name) in preop_schema_features
            if sum(1 for r in per_patient_results if name in r["present_in_window"].get(table, set())) == 0
        )
        print()
        if never_found:
            print(f"Features in the schema that NO patient has in their pre-op window: {', '.join(never_found)}")
        else:
            print("Every schema feature is present for at least one patient's pre-op window.")

    # ---------------- Departments / labels ----------------
    print()
    print("Patients per department:")
    for dept, count in sorted(dept_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {dept}: {count}")

    if any(l != "unknown" for l in label_counts):
        print()
        print("Died vs survived:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_features.py <patient_data_dir_or_file> [parameters.csv]")
        sys.exit(1)
    data_path = sys.argv[1]
    params_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(data_path, params_path)
