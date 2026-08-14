"""
Departments found: ['AN', 'CTS', 'DM', 'GS', 'IM', 'NS', 'OG', 'OL', 'OS', 'OT', 'PED', 'PS', 'RAD', 'RO', 'UR']

  AN    :     55 available -> sampled 55
  CTS   :   6555 available -> sampled 300
  DM    :      1 available -> sampled 1
  GS    :  28035 available -> sampled 300
  IM    :     60 available -> sampled 60
  NS    :   7623 available -> sampled 300
  OG    :  11253 available -> sampled 300
  OL    :   9494 available -> sampled 300
  OS    :  12223 available -> sampled 300
  OT    :  11119 available -> sampled 300
  PED   :     27 available -> sampled 27
  PS    :   3582 available -> sampled 300
  RAD   :    331 available -> sampled 300
  RO    :     11 available -> sampled 11
  UR    :   8575 available -> sampled 300

Total survived patients sampled: 3154
sample_survived_by_department.py

Run this LOCALLY on your machine (Windows), pointed at your real dataset.
No external packages needed — pure Python standard library only.

WHAT IT DOES
------------
1. Scans every JSON file in the 'survived' folder.
2. Reads each patient's department (from their last operation record).
3. Groups patients by department.
4. Samples up to TARGET_PER_DEPT patients from EACH department (if a
   department has fewer patients than that, it just takes all of them —
   it will never invent patients).
5. Copies the sampled files into a new output folder.
6. Zips that output folder into one .zip file, ready to upload here.
7. Prints a summary table (department -> how many available vs how many
   sampled) and a manifest CSV so you can see exactly what was picked.

HOW TO RUN
----------
1. Save this file anywhere on your PC, e.g. Desktop.
2. Open Command Prompt (search "cmd").
3. Run:
       cd Desktop
       python sample_survived_by_department.py
   (If "python" isn't recognized, try "py" instead of "python".)

CONFIGURATION
-------------
Just edit the three variables below if your paths differ or you want a
different sample size.
"""

import os
import json
import random
import shutil
import zipfile
import csv
from collections import defaultdict

# ---------------------------------------------------------------
# CONFIG — edit these if needed
# ---------------------------------------------------------------
SURVIVED_DIR = r"C:\Users\pc\Desktop\INSPIRE\dataset\subjects\survived"
OUTPUT_DIR = r"C:\Users\pc\Desktop\INSPIRE\dataset\survived_sampled"          # sampled JSONs copied here
OUTPUT_ZIP = r"C:\Users\pc\Desktop\INSPIRE\dataset\survived_sampled.zip"      # final zip to upload
TARGET_PER_DEPT = 300     # aim for this many survived patients per department
RANDOM_SEED = 42          # fixed seed = reproducible sample if you re-run this


def get_department(json_path):
    """Reads a patient JSON and returns their department (from the last
    operation record), or None if it can't be determined."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    operations = data.get("operations") or []
    if not operations:
        return None

    last_op = operations[-1]
    dept = last_op.get("department")
    if not dept:
        return None
    return dept


def main():
    random.seed(RANDOM_SEED)

    if not os.path.isdir(SURVIVED_DIR):
        print(f"ERROR: folder not found: {SURVIVED_DIR}")
        print("Double check the path and try again.")
        return

    all_files = [f for f in os.listdir(SURVIVED_DIR) if f.endswith(".json")]
    print(f"Found {len(all_files)} survived patient files. Reading departments "
          f"(this may take a few minutes for 90k files)...")

    dept_to_files = defaultdict(list)
    unreadable = 0
    for i, fname in enumerate(all_files, 1):
        if i % 5000 == 0:
            print(f"  ...scanned {i}/{len(all_files)}")
        path = os.path.join(SURVIVED_DIR, fname)
        dept = get_department(path)
        if dept is None:
            unreadable += 1
            continue
        dept_to_files[dept].append(fname)

    print(f"\nDone scanning. {unreadable} files skipped (no readable department).")
    print(f"Departments found: {sorted(dept_to_files.keys())}\n")

    # ---------------------------------------------------------------
    # Sample up to TARGET_PER_DEPT from each department
    # ---------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_rows = []
    summary_rows = []

    for dept in sorted(dept_to_files.keys()):
        files = dept_to_files[dept]
        n_available = len(files)
        n_sample = min(TARGET_PER_DEPT, n_available)
        sampled = random.sample(files, n_sample)

        for fname in sampled:
            src = os.path.join(SURVIVED_DIR, fname)
            dst = os.path.join(OUTPUT_DIR, fname)
            shutil.copy2(src, dst)
            manifest_rows.append({"subject_file": fname, "department": dept})

        summary_rows.append({"department": dept, "available": n_available, "sampled": n_sample})
        print(f"  {dept:6s}: {n_available:6d} available -> sampled {n_sample}")

    total_sampled = sum(r["sampled"] for r in summary_rows)
    print(f"\nTotal survived patients sampled: {total_sampled}")

    # ---------------------------------------------------------------
    # Write manifest CSV (so you know exactly which patients were picked)
    # ---------------------------------------------------------------
    manifest_path = os.path.join(OUTPUT_DIR, "_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_file", "department"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written to: {manifest_path}")

    # ---------------------------------------------------------------
    # Zip everything up
    # ---------------------------------------------------------------
    print(f"\nZipping to {OUTPUT_ZIP} ...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_DIR):
            zf.write(os.path.join(OUTPUT_DIR, fname), arcname=fname)

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"Done. Zip file size: {zip_size_mb:.1f} MB")
    print(f"Upload this file: {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
