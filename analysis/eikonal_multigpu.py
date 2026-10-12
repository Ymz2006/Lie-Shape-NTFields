"""Expand the existing frozen Eikonal sweep after authorization of GPUs 0–3."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.reproduce_eikonal.config import MODELS, ROOT, jobs, sets_for, tag
from analysis.reproduce_eikonal.report import build_report
from analysis.reproduce_eikonal.run import sha256, stamp, validate_eval, write_json

GPUS = {
    0: 'GPU-a6b99c0e-bfa5-9158-10e3-3e1a2a521990',
    1: 'GPU-9db5dcc2-e084-8245-66bf-6406a519835f',
    2: 'GPU-afc89ec0-d38c-ac03-434b-952759616727',
    3: 'GPU-ed74700f-7cd4-63be-4218-89dfd7b69248',
}


def train_worker(args):
    import torch
    expected = GPUS[args.gpu]

    def ensure_assigned_gpu():
        if os.environ.get('CUDA_VISIBLE_DEVICES') != expected or torch.cuda.device_count() != 1:
            raise RuntimeError('Worker must expose exactly its assigned physical GPU.')
        visible = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        if expected not in visible:
            raise RuntimeError('Assigned GPU UUID is unavailable.')

    original = ROOT / 'analysis/reproduce_eikonal/train_one.py'
    source = original.read_text()
    # Adapt only the device guard and provenance. Model construction, RNG seeds,
    # hyperparameters, optimizer, losses and checkpoint behavior are unchanged.
    assert source.count('ensure_gpu3()') == 1
    assert source.count("'physical_device': 3, 'gpu_uuid': GPU_UUID") == 1
    source = source.replace('ensure_gpu3()', 'ensure_assigned_gpu()')
    source = source.replace("'physical_device': 3, 'gpu_uuid': GPU_UUID",
                            "'physical_device': PHYSICAL_GPU, 'gpu_uuid': ASSIGNED_GPU_UUID")
    sys.argv = [str(original), '--cell', args.cell, '--model', args.model,
                '--seed', str(args.seed), '--out', str(args.out), '--epochs', str(args.epochs)]
    namespace = {'__name__': '__main__', '__file__': str(original),
                 'ensure_assigned_gpu': ensure_assigned_gpu,
                 'PHYSICAL_GPU': args.gpu, 'ASSIGNED_GPU_UUID': expected}
    exec(compile(source, str(original), 'exec'), namespace)
    path = args.out / tag(args.cell, args.model, args.seed) / 'training.json'
    metadata = json.loads(path.read_text())
    metadata['device_adapter_sha256'] = sha256(Path(__file__))
    write_json(path, metadata)


def metadata_at(out, name):
    path = out / 'checkpoints' / name / 'training.json'
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def validate_expansion(args):
    import torch
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[0], PYTHONDONTWRITEBYTECODE='1',
               NTF_NO_PLOT='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-u', str(Path(__file__)), '--train-worker', '--gpu', '0',
               '--cell', 'Tshape3d_env1', '--model', 'iso', '--seed', '1', '--epochs', '2',
               '--out', str(out / 'checkpoints')]
    subprocess.run(command, env=env, check=True, cwd=ROOT)
    name = tag('Tshape3d_env1', 'iso', 1)
    old_root = ROOT / 'results/eikonal_five_seed_device3_smoke'
    previous = next((old_root / 'checkpoints' / name).glob('Model_Epoch_00002_*.pt'))
    current = next((out / 'checkpoints' / name).glob('Model_Epoch_00002_*.pt'))
    a, b = (torch.load(path, map_location='cpu') for path in (previous, current))
    assert torch.equal(a['B_state_dict'], b['B_state_dict'])
    assert all(torch.equal(value, b['model_state_dict'][key]) for key, value in a['model_state_dict'].items())
    result_path = out / 'evaluation.json'
    subprocess.run([sys.executable, '-u', 'analysis/plan_eval.py', '--kind', 'lie_iso',
                    '--checkpoint', str(current), '--dataPath', 'testing_data/3dshape/Tshape3d_env1',
                    '--out', str(result_path), '--device', 'cuda:0', '--batch', '125',
                    '--workers', '2', '--steps', '200', '--seed', '0', '--cases', '10'],
                   env=env, check=True, cwd=ROOT)
    old = json.loads((old_root / 'plan' / name / 'Tshape3d_env1.json').read_text())
    new = json.loads(result_path.read_text())
    assert old['records'] == new['records']
    for field in ('success_rate', 'collision_rate', 'nonconv_rate'):
        assert old['summary'][field] == new['summary'][field]
    print('PASS: GPU 0 reproduces GPU 3 trained weights bitwise; two collision workers reproduce all ten per-case records exactly.', flush=True)
    (out / 'VALIDATION.txt').write_text('PASS: GPU 0/3 weights bitwise identical; two/eight collision workers yield identical per-case records.\n')


def controller(args):
    import numpy as np
    import torch
    out = args.out.resolve()
    os.chdir(ROOT)
    lock = (out / '.multigpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    status_path = out / 'status.json'
    status = json.loads(status_path.read_text())
    legacy = json.loads((out / 'expansion/previous_status.json').read_text())
    reserved = {name for name, value in legacy['jobs'].items() if value['state'] == 'training'}
    assert len(reserved) == 4, f'Unexpected original jobs: {reserved}'
    assert all(v['state'] in ('training', 'complete') for v in legacy['jobs'].values())
    assert torch.__version__ == manifest['torch'] and torch.version.cuda == manifest['cuda']
    assert np.__version__ == manifest['numpy'] and sys.version == manifest['python']
    display_path = 'analysis/reproduce_eikonal/report.py'
    changes = []
    for relative, expected in manifest['hashes'].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            if relative != display_path:
                raise RuntimeError(f'Frozen training/input source changed: {relative}')
            changes.append({'path': relative, 'original_sha256': expected, 'current_sha256': actual,
                            'reason': 'Report now documents the authorized GPU expansion.'})
    update = {'at': stamp(), 'authorized_physical_devices': list(GPUS), 'gpu_uuids': GPUS,
              'device_adapter_sha256': sha256(Path(__file__)), 'source_changes': changes,
              'note': 'User authorized GPUs 0–2 in addition to device 3. Preserve all existing work.',
              'collision_workers': 2, 'parallel_per_gpu': 2,
              'pending_legacy_training': sorted(reserved)}
    manifest.setdefault('execution_updates', []).append(update)
    manifest['physical_devices'] = list(GPUS)
    manifest['execution_device_map'] = {str(k): v for k, v in GPUS.items()}
    write_json(manifest_path, manifest)
    snapshot = out / 'expansion/eikonal_multigpu.py'
    snapshot.write_text(Path(__file__).read_text())
    (out / 'expansion/report_multigpu.py').write_text((ROOT / display_path).read_text())
    status.update(state='running', physical_devices=list(GPUS), controller='four GPU expansion',
                  updated=stamp(), expanded_at=stamp(), parallel_per_gpu=2)
    write_json(status_path, status)
    build_report(out)
    mutex = threading.Lock()
    tokens = queue.Queue()
    for gpu in (0, 1, 2):
        for _ in range(2):
            tokens.put(gpu)
    report_time = [time.monotonic()]

    def update_status(name, **fields):
        with mutex:
            status['jobs'].setdefault(name, {}).update(fields)
            status['updated'] = stamp()
            write_json(status_path, status)
            if fields.get('state') in ('complete', 'failed') or time.monotonic() - report_time[0] > 60:
                build_report(out)
                report_time[0] = time.monotonic()
        print(stamp(), name, fields, flush=True)

    def enable_gpu3():
        while not all(metadata_at(out, name).get('completed') for name in reserved):
            time.sleep(5)
        for _ in range(2):
            tokens.put(3)
        with mutex:
            status['legacy_training_drained'] = stamp()
            write_json(status_path, status)
        print(stamp(), 'Preserved training on GPU 3 finished; GPU 3 added to the new queue.', flush=True)

    threading.Thread(target=enable_gpu3, daemon=True).start()

    def execute(command, log, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu], OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NTF_NO_PLOT='1',
                   PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/eikonal-matplotlib')
        with log.open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, check=True)

    def work(job):
        cell, model, seed = job
        name = tag(*job)
        gpu = None
        try:
            if name in reserved:
                while not metadata_at(out, name).get('completed'):
                    time.sleep(5)
            gpu = tokens.get()
            metadata = metadata_at(out, name)
            if not metadata.get('completed'):
                update_status(name, state='training', cell=cell, model=model, seed=seed,
                              physical_device=gpu, gpu_uuid=GPUS[gpu], started=stamp())
                execute([sys.executable, '-u', str(Path(__file__)), '--train-worker', '--gpu', str(gpu),
                         '--cell', cell, '--model', model, '--seed', str(seed),
                         '--out', str(out / 'checkpoints'), '--epochs', '4000'],
                        out / 'logs' / (name + '.train.log'), gpu)
                metadata = metadata_at(out, name)
            assert metadata['epochs'] == 4000 and metadata['seed'] == seed and metadata['completed']
            checkpoint = Path(metadata['checkpoint'])
            checkpoint_hash = sha256(checkpoint)
            destination = out / 'plan' / name
            destination.mkdir(parents=True, exist_ok=True)
            for i, (dataset, _) in enumerate(sets_for(cell)):
                target = destination / (dataset + '.json')
                if target.exists():
                    data = validate_eval(target, job, dataset, checkpoint, manifest['set_sizes'][dataset])
                    assert data['reproduction']['checkpoint_sha256'] == checkpoint_hash
                    continue
                update_status(name, state='evaluating', checkpoint=str(checkpoint), completed_sets=i,
                              physical_device=gpu, gpu_uuid=GPUS[gpu], current_set=dataset)
                command = [sys.executable, '-u', 'analysis/plan_eval.py', '--kind', MODELS[model][1],
                           '--checkpoint', str(checkpoint), '--dataPath', f'testing_data/3dshape/{dataset}',
                           '--out', str(target), '--device', 'cuda:0', '--batch', '125',
                           '--workers', '2', '--steps', '200', '--seed', '0', '--planner', 'mppi']
                execute(command, out / 'logs' / (name + '.' + dataset + '.log'), gpu)
                data = validate_eval(target, job, dataset, checkpoint, manifest['set_sizes'][dataset])
                data['reproduction'] = {'training_seed': seed, 'physical_device': gpu,
                                        'gpu_uuid': GPUS[gpu], 'training_physical_device': metadata['physical_device'],
                                        'checkpoint_sha256': checkpoint_hash, 'command': command,
                                        'device_adapter_sha256': sha256(Path(__file__)), 'completed': stamp()}
                write_json(target, data)
            update_status(name, state='complete', completed_sets=len(sets_for(cell)), finished=stamp())
        except Exception as exc:
            update_status(name, state='failed', error=str(exc), finished=stamp())
            raise
        finally:
            if gpu is not None:
                tokens.put(gpu)

    pending = []
    for job in jobs():
        name = tag(*job)
        if status['jobs'].get(name, {}).get('state') != 'complete':
            pending.append(job)
            continue
        metadata = metadata_at(out, name)
        checkpoint = Path(metadata['checkpoint'])
        digest = sha256(checkpoint)
        for dataset, _ in sets_for(job[0]):
            data = validate_eval(out / 'plan' / name / (dataset + '.json'), job, dataset,
                                 checkpoint, manifest['set_sizes'][dataset])
            assert data['reproduction']['checkpoint_sha256'] == digest
    print(stamp(), f'{90-len(pending)} completed jobs validated; {len(pending)} remaining.', flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(work, job): job for job in pending}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], str(exc)))
    status.update(state='failed' if failures else 'complete', finished=stamp(), updated=stamp())
    write_json(status_path, status)
    build_report(out)
    if failures:
        raise RuntimeError(str(failures))
    print('All 90 jobs and 570 evaluations complete.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'results/eikonal_five_seed_device3')
    parser.add_argument('--train-worker', action='store_true')
    parser.add_argument('--validate-expansion', action='store_true')
    parser.add_argument('--gpu', type=int, choices=GPUS)
    parser.add_argument('--cell')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int, default=4000)
    args = parser.parse_args()
    if args.validate_expansion:
        validate_expansion(args)
    elif args.train_worker:
        train_worker(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()
