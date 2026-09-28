"""
Build the Kaggle/Colab notebook from the stage files in pipeline/ (single source of truth).

    python build_notebook.py                 # -> ../src/INSPIRE_Multimodal_Mortality_DNN_v2.ipynb
    python build_notebook.py --out my.ipynb

Stage files use the "percent" cell format (the same one Jupytext/VS Code/Spyder use):
    # %%              -> a code cell
    # %% [markdown]   -> a markdown cell (every line prefixed with "# ")

A code cell whose first line is  `# INLINE: inspire_dnn/<module>.py`  is replaced by that
module's full source when the notebook is built, so the notebook stays self-contained
(no imports from this repo needed on Kaggle) while the pure, data-free components
(NEWS2 scoring, the model layers) live in ONE place and are unit-tested in tests/.

`python build_notebook.py --check` rebuilds in memory and fails if the committed notebook
has drifted from the stage files -- run it before committing either.
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(HERE, "pipeline")
NOTEBOOK_DIR = os.path.join(HERE, "notebooks")
PROFILES = {
    "full": ("INSPIRE_DNN_v3_full_cohort.ipynb",
             "**FULL-COHORT notebook** (`RUN_PROFILE = 'full'`): all ~99,886 patients, every analysis "
             "switched on (ablations, 5-model ensemble, learning curve, imputation benchmarks), no "
             "run-time or memory budget. Needs `subjects.zip` (the full cohort) in the root of My Drive. "
             "Expect several hours on a single GPU -- the parsed data and training checkpoints are "
             "saved to Drive, so a disconnected session resumes rather than restarts."),
    "subset": ("INSPIRE_DNN_v3_subset_analysis.ipynb",
               "**SUBSET notebook** (`RUN_PROFILE = 'subset'`): for analysing the pipeline on "
               "`subjects_sample.zip` (every real death + a random sample of survivors). Same code as "
               "the full notebook; smaller ensemble (3) and shorter training caps."),
}
INLINE_RE = re.compile(r"^#\s*INLINE:\s*(\S+)\s*$")


def stage_files():
    return sorted(f for f in os.listdir(PIPELINE_DIR) if re.match(r"^\d\d_.*\.py$", f))


def parse_percent(text):
    cells, cur_type, cur_lines = [], None, []
    for line in text.split("\n"):
        if line.startswith("# %%"):
            if cur_type is not None:
                cells.append((cur_type, cur_lines))
            cur_type = "markdown" if "[markdown]" in line else "code"
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_type is not None:
        cells.append((cur_type, cur_lines))
    out = []
    for ctype, lines in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if ctype == "markdown":
            lines = [l[2:] if l.startswith("# ") else ("" if l == "#" else l) for l in lines]
        else:
            m = INLINE_RE.match(lines[0]) if lines else None
            if m:
                with open(os.path.join(HERE, m.group(1))) as f:
                    lines = [f"# --- inlined from src-dnn/{m.group(1)} (edit it there, then rebuild) ---"] + \
                            f.read().rstrip("\n").split("\n")
        out.append((ctype, "\n".join(lines)))
    return out


def build(profile):
    nb_cells = []
    for fname in stage_files():
        with open(os.path.join(PIPELINE_DIR, fname)) as f:
            for ctype, src in parse_percent(f.read()):
                src = src.replace('RUN_PROFILE = "subset"', f'RUN_PROFILE = "{profile}"')
                src = src.replace("PROFILE_BANNER", PROFILES[profile][1])
                lines = src.split("\n")
                source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                cell = {"cell_type": ctype, "id": f"c{len(nb_cells):04d}", "metadata": {}, "source": source}
                if ctype == "code":
                    cell.update({"execution_count": None, "outputs": []})
                nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed notebooks differ from pipeline/")
    args = ap.parse_args()
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    drift = False
    for profile, (fname, _) in PROFILES.items():
        nb = build(profile)
        text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
        out = os.path.join(NOTEBOOK_DIR, fname)
        if args.check:
            same = os.path.exists(out) and open(out).read() == text
            print(f"{fname}: {'in sync' if same else 'DRIFT -- rebuild'}")
            drift |= not same
            continue
        with open(out, "w") as f:
            f.write(text)
        n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
        print(f"Wrote notebooks/{fname}: {len(nb['cells'])} cells ({n_code} code), RUN_PROFILE={profile!r}")
    sys.exit(1 if drift else 0)
