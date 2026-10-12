"""Run the full fresh sweep, resume completed work, and regenerate the report."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.reproduce_eikonal.config import CELLS, GPU_UUID, MODELS, ROOT, ensure_gpu3, jobs, sets_for, tag
from analysis.reproduce_eikonal.report import build_report

LOCK = threading.Lock()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_eval(path, job, dataset, checkpoint, expected_n):
    data = json.loads(path.read_text())
    sm, records = data['summary'], data['records']
    cell, model, seed = job
    assert sm['kind'] == MODELS[model][1]
    assert Path(sm['checkpoint']).resolve() == checkpoint.resolve()
    assert sm['seed'] == 0 and sm['batch'] == 125 and sm['steps'] == 200
    assert sm['planner'] == 'mppi' and not sm['group_conv']
    assert sm['n_cases'] == expected_n and sm['n_endpoints_ok'] == len(records) > 0
    assert len({row['case'] for row in records}) == len(records)
    assert all(row['success'] == (row['converged'] and not row['collision']) for row in records)
    for key, field in [('success_rate', 'success'), ('collision_rate', 'collision')]:
        assert abs(sm[key] - sum(row[field] for row in records) / len(records)) < 1e-12
    assert abs(sm['nonconv_rate'] - sum(not row['converged'] for row in records) / len(records)) < 1e-12
    assert Path(sm['dataPath']).name == dataset
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'results/eikonal_five_seed_device3')
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--smoke', action='store_true', help='2 epochs / 10 cases; not benchmark results')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        build_report(out)
        return
    lock_stream = (out / '.run.lock').open('a')
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A sweep is already running in this output directory.')
    import numpy as np
    import torch
    ensure_gpu3()
    if args.parallel < 1 or args.parallel > 4:
        raise ValueError('Use 1–4 concurrent processes on the one permitted GPU.')
    all_jobs = jobs() if not args.smoke else [(next(iter(CELLS)), m, 1) for m in MODELS]
    epochs = 2 if args.smoke else 4000
    status_path = out / 'status.json'
    status = {'started': stamp(), 'updated': stamp(), 'state': 'running',
              'smoke': args.smoke, 'epochs': epochs, 'physical_device': 3,
              'gpu_uuid': GPU_UUID, 'jobs_total': len(all_jobs), 'jobs': {}}
    last_report = [0.0]
    source_files = sorted({p for pkg, *_ in MODELS.values() for p in (ROOT / 'models' / pkg).glob('*.py')})
    source_files += [ROOT / 'models/data_mlp.py', ROOT / 'models/rnl.py',
                     ROOT / 'analysis/plan_eval.py', ROOT / 'analysis/common.py',
                     ROOT / 'dataprocessing/preprocess_obj.py']
    source_files += sorted((ROOT / 'analysis/reproduce_eikonal').glob('*.py'))
    data_files = [ROOT / 'Isotropic vs Anisotropic Lie Eikonal.html']
    set_sizes = {}
    for cell in CELLS:
        data_files.extend(sorted((ROOT / 'datasets/3dshape' / cell).glob('*.npy')))
        data_files.append(ROOT / 'datasets/3dshape' / cell / 'meta.json')
        for dataset, _ in sets_for(cell):
            folder = ROOT / 'testing_data/3dshape' / dataset
            data_files.extend((folder / 'sampled_points.npy', folder / 'meta.json'))
            set_sizes[dataset] = len(np.load(folder / 'sampled_points.npy', mmap_mode='r'))
            meta = json.loads((folder / 'meta.json').read_text())
            for key in ('shape_obj', 'env_obj'):
                candidate = ROOT / meta[key]
                if not candidate.exists():
                    candidate = ROOT / 'datasets/3dshape' / Path(meta[key]).name
                data_files.append(candidate)
    data_files = sorted(set(data_files))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files + data_files}
    source_hashes = {str(p.relative_to(ROOT)): hashes[str(p.relative_to(ROOT))] for p in source_files}
    manifest = {'created': stamp(), 'physical_device': 3, 'gpu_uuid': GPU_UUID,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'numpy': np.__version__, 'python': sys.version, 'epochs': epochs,
                'seeds': [1] if args.smoke else [1, 2, 3, 4, 5], 'evaluation_seed': 0,
                'parallel': args.parallel, 'models': MODELS, 'set_sizes': set_sizes,
                'hashes': hashes, 'fresh_training': True,
                'evaluation': {'planner': 'mppi', 'batch': 125, 'workers': 8, 'steps': 200,
                               'step': .015, 'temperature': 50, 'samples': 50, 'horizon': 5,
                               'group_conv': False},
                'checkpoint_policy': 'Only final checkpoint; periodic serialization and cosmetic plots omitted.'}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ('hashes', 'epochs', 'seeds', 'models', 'gpu_uuid', 'torch', 'cuda', 'numpy', 'python'):
            if old[key] != json.loads(json.dumps(manifest[key])):
                raise RuntimeError(f'Resume refused: manifest {key} changed.')
    else:
        write_json(manifest_path, manifest)
        packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                  capture_output=True, text=True, check=True)
        (out / 'environment.txt').write_text(packages.stdout)
        for p in source_files:
            target = out / 'source' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        shutil.copy2(ROOT / 'Isotropic vs Anisotropic Lie Eikonal.html', out / 'original.html')
    if not args.smoke and shutil.disk_usage(out).free < 4 * 1024**3:
        raise RuntimeError('At least 4 GiB free needed for the fresh sweep.')

    def update(name, **fields):
        with LOCK:
            status['jobs'].setdefault(name, {}).update(fields)
            status['updated'] = stamp()
            write_json(status_path, status)
            if fields.get('state') in ('complete', 'failed') or time.monotonic() - last_report[0] >= 60:
                build_report(out)
                last_report[0] = time.monotonic()
        print(stamp(), name, fields, flush=True)

    def execute(command, log):
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NTF_NO_PLOT='1', MPLCONFIGDIR='/tmp/eikonal-matplotlib',
                   PYTHONDONTWRITEBYTECODE='1')
        with log.open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)

    def work(job):
        cell, model, seed = job
        name = tag(*job)
        try:
            for relative, expected in source_hashes.items():
                if sha256(ROOT / relative) != expected:
                    raise RuntimeError(f'Source changed during sweep: {relative}')
            training_root = out / 'checkpoints'
            run = training_root / name
            metadata_path = run / 'training.json'
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            checkpoint = Path(metadata['checkpoint']) if metadata.get('completed') else None
            if checkpoint is None or not checkpoint.exists():
                update(name, state='training', seed=seed, cell=cell, model=model, started=stamp())
                log = out / 'logs' / (name + '.train.log')
                log.parent.mkdir(exist_ok=True)
                execute([sys.executable, '-u', 'analysis/reproduce_eikonal/train_one.py',
                         '--cell', cell, '--model', model, '--seed', str(seed),
                         '--out', str(training_root), '--epochs', str(epochs)], log)
                metadata = json.loads(metadata_path.read_text())
                checkpoint = Path(metadata['checkpoint'])
            checkpoint_hash = sha256(checkpoint)
            sets = sets_for(cell)[:1] if args.smoke else sets_for(cell)
            destination = out / 'plan' / name
            destination.mkdir(parents=True, exist_ok=True)
            for i, (dataset, _) in enumerate(sets):
                target = destination / (dataset + '.json')
                expected_n = min(10, set_sizes[dataset]) if args.smoke else set_sizes[dataset]
                if target.exists():
                    data = validate_eval(target, job, dataset, checkpoint, expected_n)
                    assert data['reproduction']['checkpoint_sha256'] == checkpoint_hash
                    continue
                update(name, state='evaluating', checkpoint=str(checkpoint), completed_sets=i, current_set=dataset)
                command = [sys.executable, '-u', 'analysis/plan_eval.py', '--kind', MODELS[model][1],
                           '--checkpoint', str(checkpoint), '--dataPath', f'testing_data/3dshape/{dataset}',
                           '--out', str(target), '--device', 'cuda:0', '--batch', '125',
                           '--workers', '8', '--steps', '200', '--seed', '0', '--planner', 'mppi']
                if args.smoke:
                    command.extend(('--cases', '10'))
                execute(command, out / 'logs' / (name + '.' + dataset + '.log'))
                data = validate_eval(target, job, dataset, checkpoint, expected_n)
                data['reproduction'] = {'training_seed': seed, 'physical_device': 3,
                                        'gpu_uuid': GPU_UUID, 'checkpoint_sha256': checkpoint_hash,
                                        'command': command, 'completed': stamp()}
                write_json(target, data)
            update(name, state='complete', completed_sets=len(sets), finished=stamp())
        except Exception as exc:
            update(name, state='failed', error=str(exc), finished=stamp())
            raise

    write_json(status_path, status)
    build_report(out)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(work, job): job for job in all_jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], str(exc)))
    status.update(state='failed' if failures else 'complete', finished=stamp(), updated=stamp())
    write_json(status_path, status)
    build_report(out)
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures}')
    print(f'Completed {len(all_jobs)} jobs. Report: {out / "REPORT.html"}', flush=True)


if __name__ == '__main__':
    main()
