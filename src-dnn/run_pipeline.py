"""
Run the pipeline stages top to bottom in ONE shared namespace -- exactly how the notebook
runs (every cell sees every earlier cell's variables), without the fragile
`from previous_stage import *` chaining the earlier conversion used (which silently drops
every `_underscore` name, e.g. `_HOLD_IDS`, and breaks on later re-assignments).

    python run_pipeline.py                                   # all stages
    python run_pipeline.py --until 07                        # stop after the sampling stage
    python run_pipeline.py --subjects-dir /data/subjects \\
        --config '{"MAX_SUBJECTS_PER_CLASS": 2000, "EPOCHS_FINETUNE": 20}'

--config overrides CONFIG keys after CONFIG is built (JSON; null -> None). It is passed via
the INSPIRE_CONFIG_OVERRIDES environment variable, which the notebook honours too.
Figures are saved to --fig-dir instead of shown when running headless.
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=None, help="last stage number to run, e.g. 07")
    ap.add_argument("--subjects-dir", default=None)
    ap.add_argument("--config", default=None, help="JSON dict of CONFIG overrides")
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "run_figures"))
    args = ap.parse_args()

    overrides = json.loads(args.config) if args.config else {}
    if args.subjects_dir:
        overrides["SUBJECTS_DIR"] = args.subjects_dir
    if overrides:
        os.environ["INSPIRE_CONFIG_OVERRIDES"] = json.dumps(overrides)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.fig_dir, exist_ok=True)
    fig_counter = {"n": 0}

    def _save_instead_of_show(*a, **k):
        for num in plt.get_fignums():
            fig_counter["n"] += 1
            plt.figure(num).savefig(os.path.join(args.fig_dir, f"fig_{fig_counter['n']:03d}.png"), dpi=80)
        plt.close("all")
    plt.show = _save_instead_of_show

    sys.path.insert(0, HERE)   # so `from inspire_dnn... import *` resolves (the notebook inlines these instead)
    stages = sorted(f for f in os.listdir(os.path.join(HERE, "pipeline")) if re.match(r"^\d\d_.*\.py$", f))
    if args.until:
        stages = [s for s in stages if s[:2] <= args.until.zfill(2)]

    ns = {"__name__": "__main__"}
    t_all = time.time()
    for stage in stages:
        path = os.path.join(HERE, "pipeline", stage)
        print(f"\n{'=' * 78}\n>>> {stage}\n{'=' * 78}", flush=True)
        t0 = time.time()
        with open(path) as f:
            code = compile(f.read(), path, "exec")
        exec(code, ns)
        print(f"<<< {stage} done in {time.time() - t0:.1f}s", flush=True)
    print(f"\nAll {len(stages)} stages finished in {time.time() - t_all:.1f}s. Figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
