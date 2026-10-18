"""Run one isolated experiment. Repeats require distinct --training-seed values.

python run_pipeline.py --pooling mean --training-seed 101 --split-seed 42 \
    --subjects-dir /path/to/subjects --run-dir experiments/mean_101

A fresh run never loads old checkpoints. --resume requires the original --run-dir
and identical arguments, code, environment and input data. It resumes the last saved
epoch boundary. --until 10 is sufficient for primary evaluation; later stages are
optional analyses. For single-model comparisons pass --config '{"N_ENSEMBLE": 1}'.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

from inspire_dnn.experiments import (SCHEMA_VERSION, Tee, atomic_json, object_hash,
    source_provenance, environment_info, dataset_provenance, export_evaluation,
    run_lock, utc_now)

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--until', choices=[f'{n:02d}' for n in range(14)])
    ap.add_argument('--subjects-dir')
    ap.add_argument('--config', help='JSON object of CONFIG overrides')
    ap.add_argument('--profile', default='subset', choices=['subset', 'full', 'isambard'])
    ap.add_argument('--pooling', required=True, choices=['mean', 'attention'])
    ap.add_argument('--training-seed', required=True, type=int)
    ap.add_argument('--split-seed', type=int, default=42)
    ap.add_argument('--data-seed', type=int, default=42, help='fixed cohort sampling/preprocessing seed')
    ap.add_argument('--bootstrap-seed', type=int, default=2026)
    ap.add_argument('--split-file', help='reuse a saved splits.json; checked against cohort and settings')
    ap.add_argument('--run-dir', help='new directory; generated under experiments/ if omitted')
    ap.add_argument('--resume', action='store_true', help='explicitly continue an interrupted run')
    args = ap.parse_args(argv)
    for name in ('training_seed','split_seed','data_seed','bootstrap_seed'):
        if not 0 <= getattr(args, name) < 2**32:
            ap.error(f'--{name.replace("_", "-")} must be in [0, 2**32)')
    if args.resume and not args.run_dir:
        ap.error('--resume requires --run-dir')
    overrides = json.loads(os.environ.get('INSPIRE_CONFIG_OVERRIDES', '{}') or '{}')
    overrides.update(json.loads(args.config) if args.config else {})
    # Paths are exclusively run-owned; no shared checkpoints/tables/figures.
    for name in ('OUTPUT_DIR','CHECKPOINT_DIR','RESUME'):
        if name in overrides:
            ap.error(f'{name} is managed by --run-dir/--resume, not --config')
    explicit = dict(POOLING=args.pooling, TRAINING_SEED=args.training_seed,
                    SPLIT_SEED=args.split_seed, DATA_SEED=args.data_seed,
                    BOOTSTRAP_SEED=args.bootstrap_seed)
    for key, value in explicit.items():
        if key in overrides and overrides[key] != value:
            ap.error(f'{key} conflicts with its command-line argument')
    overrides.update(explicit)
    if args.subjects_dir:
        overrides['SUBJECTS_DIR'] = str(Path(args.subjects_dir).resolve())
    if args.split_file:
        overrides['SPLIT_FILE'] = str(Path(args.split_file).resolve())
    return args, overrides


def main(argv=None):
    args, overrides = parse_args(argv)
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (
        HERE / 'experiments' / f'{utc_now().replace(":", "-")}_{args.pooling}_s{args.training_seed}_{uuid.uuid4().hex[:8]}')
    if args.resume:
        if not (run_dir / 'results.json').exists():
            raise ValueError('Cannot resume: results.json is missing')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    source = source_provenance(HERE)
    request = {'profile': args.profile, 'overrides': overrides}
    request_hash = object_hash(request)
    with run_lock(run_dir):
        old = json.loads((run_dir / 'results.json').read_text()) if args.resume else None
        if old:
            if old['status'] == 'completed':
                raise ValueError('This run is completed; create a new run with a new training seed instead')
            if old['request_sha256'] != request_hash or old['source']['source_sha256'] != source['source_sha256']:
                raise ValueError('Resume requires the original configuration and source code')
        results = json.loads(json.dumps(old)) if old else {'schema_version': SCHEMA_VERSION, 'run_id': run_dir.name,
            'started_at': utc_now(), 'request': request, 'request_sha256': request_hash,
            'source': source, 'seeds': {'training': args.training_seed, 'split': args.split_seed,
                'data': args.data_seed, 'bootstrap': args.bootstrap_seed}, 'attempts': []}
        results['status'] = 'running'
        results.pop('error', None)
        results['attempts'].append({'started_at': utc_now(), 'resume': args.resume})
        atomic_json(run_dir / 'results.json', results)
        overrides = dict(overrides, OUTPUT_DIR=str(run_dir), CHECKPOINT_DIR=str(run_dir / 'checkpoints'), RESUME=args.resume)
        os.environ['INSPIRE_CONFIG_OVERRIDES'] = json.dumps(overrides)
        os.environ['INSPIRE_HEADLESS'] = '1'
        os.environ['INSPIRE_FIG_DIR'] = str(run_dir / 'figures')
        os.environ['INSPIRE_RUN_PROFILE'] = args.profile
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        ns = {'__name__': '__main__'}
        evaluated_this_attempt = False
        started = time.monotonic()
        with (run_dir / 'terminal.log').open('a', buffering=1) as log, \
             contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import matplotlib
                matplotlib.use('Agg')
                print(f'Experiment directory: {run_dir}', flush=True)
                for path in sorted((HERE / 'pipeline').glob('[0-9][0-9]_*.py')):
                    if args.until and path.name[:2] > args.until:
                        continue
                    print(f'\n>>> {path.name}', flush=True)
                    # Shared parse caches are safe even when separate runs start together.
                    cache_guard = run_lock(ns['CONFIG']['CACHE_DIR'], wait=True) if path.name[:2] == '01' else contextlib.nullcontext()
                    with cache_guard:
                        exec(compile(path.read_text(), str(path), 'exec'), ns)
                    stage = path.name[:2]
                    if stage == '00':
                        cfg = ns['CONFIG']
                        env = environment_info()
                        if old and old.get('environment') and old['environment'] != env:
                            raise ValueError('Resume environment differs from the original run')
                        print('Fingerprinting selected input files...', flush=True)
                        data = dataset_provenance(cfg['SUBJECTS_DIR'], cfg['MAX_SUBJECTS_PER_CLASS'], cfg['DATA_SEED'])
                        if old and old.get('dataset_sha256') and old['dataset_sha256'] != data['sha256']:
                            raise ValueError('Resume input data differs from the original run')
                        results.update(config=dict(cfg), environment=env, dataset_sha256=data['sha256'])
                        ns['DATASET_FINGERPRINT'] = data['sha256']
                        ns['SOURCE_FINGERPRINT'] = source['source_sha256']
                        atomic_json(run_dir / 'dataset.json', data)
                    if stage == '02':
                        results['split_sha256'] = object_hash(ns['EXPERIMENT_SPLITS'])
                        if old and old.get('split_sha256') and old['split_sha256'] != results['split_sha256']:
                            raise ValueError('Resume split differs from original run')
                    if stage in ('09', '10'):
                        results.update(export_evaluation(ns, run_dir))
                        evaluated_this_attempt = True
                        results['architecture_fingerprint'] = ns['ARCHITECTURE_FINGERPRINT']
                    results['last_completed_stage'] = path.name
                    atomic_json(run_dir / 'results.json', results)
                results['status'] = 'completed' if evaluated_this_attempt else 'partial'
            except BaseException as exc:
                results['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
                results['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                traceback.print_exc()
                raise
            finally:
                results['updated_at'] = utc_now()
                results['attempts'][-1]['elapsed_seconds'] = time.monotonic() - started
                results['elapsed_seconds'] = sum(a.get('elapsed_seconds', 0) for a in results['attempts'])
                atomic_json(run_dir / 'results.json', results)
            print(f"Saved {results['status']} experiment: {run_dir / 'results.json'}", flush=True)


if __name__ == '__main__':
    main()
