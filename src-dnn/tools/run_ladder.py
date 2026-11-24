"""Cross-learning ablation ladder on the preprint subset (A / B / C), with saved predictions.

Runs pipeline stages 00-09 exactly as the preprint's subset experiment (EXPERIMENT1.txt:
MAX_SUBJECTS_PER_CLASS=10000, BATCH_SIZE=256, imputation benchmarks and fusion ablation
skipped), then trains three variants with the SAME recipe and seed (train_full_model, seed 42):

  A  isolated organs      coupling off, arbitration off, whole-patient off
  B  + cross-learning     coupling on,  arbitration off, whole-patient off
  C  full model           coupling on,  arbitration on,  whole-patient on

Each variant's test predictions are written to <out>/pred_<variant>.csv (subject_id, label,
probability) for paired comparison (see tools/ladder_compare.py).

    python tools/run_ladder.py --subjects-dir C:/.../INSPIRE/dataset/subjects --out <dir>
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src-dnn/

VARIANTS = {
    "A_isolated_organs": dict(coupling_mode="none", use_arbitration=False, use_whole_patient=False),
    "B_plus_cross_learning": dict(coupling_mode="learned", use_arbitration=False, use_whole_patient=False),
    "C_full_model": dict(coupling_mode="learned", use_arbitration=True, use_whole_patient=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset of " + ",".join(VARIANTS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Preprint subset settings (EXPERIMENT1.txt) + run-local cache/checkpoint folders.
    overrides = {"MAX_SUBJECTS_PER_CLASS": 10000, "BATCH_SIZE": 256,
                 "SKIP_IMPUTATION_ACCURACY_BENCHMARKS": True, "SKIP_FUSION_ABLATION": True,
                 "SUBJECTS_DIR": args.subjects_dir,
                 "CACHE_DIR": os.path.join(args.out, "parse_cache"),
                 "CHECKPOINT_DIR": os.path.join(args.out, "checkpoints")}
    os.environ["INSPIRE_CONFIG_OVERRIDES"] = json.dumps(overrides)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")

    sys.path.insert(0, HERE)
    stages = sorted(f for f in os.listdir(os.path.join(HERE, "pipeline"))
                    if re.match(r"^\d\d_.*\.py$", f) and f[:2] <= "09")
    ns = {"__name__": "__main__"}
    t_all = time.time()
    for stage in stages:
        path = os.path.join(HERE, "pipeline", stage)
        print(f"\n>>> {stage}", flush=True)
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            exec(compile(f.read(), path, "exec"), ns)
        print(f"<<< {stage} done in {time.time() - t0:.1f}s", flush=True)

    import pandas as pd
    test_ids = list(ns["TEST_IDS"])
    meta = {"n_train_items": len(ns["train_dataset"]), "n_test": len(test_ids),
            "test_deaths": int(sum(ns["PATIENT_BUNDLE"][s]["label"] for s in test_ids)),
            "seed": ns["SEED"], "config_overrides": overrides, "variants": {}}
    print(f"\nTest set: {meta['n_test']} patients, {meta['test_deaths']} deaths", flush=True)

    for name in args.variants.split(","):
        kw = VARIANTS[name]
        t0 = time.time()
        print(f"\n### training {name} {kw}", flush=True)
        m, _ = ns["train_full_model"](seed=ns["SEED"], **kw)
        r = ns["evaluate"](m, ns["test_loader"])
        pd.DataFrame({"subject_id": test_ids, "label": r["labels"].astype(int), "probability": r["probs"]}) \
            .to_csv(os.path.join(args.out, f"pred_{name}.csv"), index=False)
        n_params = sum(p.numel() for p in m.parameters())
        meta["variants"][name] = {"settings": kw, "test_auroc": float(r["auroc"]), "test_auprc": float(r["auprc"]),
                                  "n_params": n_params, "minutes": (time.time() - t0) / 60}
        print(f"    {name}: AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  params {n_params:,}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        with open(os.path.join(args.out, "ladder_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    print(f"\nAll done in {(time.time() - t_all) / 60:.1f} min. Outputs in {args.out}")


if __name__ == "__main__":
    main()
