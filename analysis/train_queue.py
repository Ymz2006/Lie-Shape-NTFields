"""GPU-aware queue: train the comparison models (metric / shape / lie, metric
architecture, Eikonal only, 4000 epochs) and chain each model's planning suite.

    python analysis/train_queue.py            # inside the container, cwd /workspace

A job starts when one of --gpus has at least --need MB free (checked with
nvidia-smi), pinned to that GPU with CUDA_VISIBLE_DEVICES; after launching it waits
--settle seconds so the job's allocation shows up before the next decision.
"""
import os
import time
import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--gpus', default='0,1,2')
ap.add_argument('--need', type=int, default=1200, help='MB free required to start a shape/lie job')
ap.add_argument('--need_metric', type=int, default=2300,
                help='metric_baseline renders a speed plot every 500 epochs (~1.7 GB peak)')
ap.add_argument('--settle', type=int, default=150)
ap.add_argument('--max_jobs', type=int, default=6)
ap.add_argument('--jobs', default='', help='comma list of job names to restrict to')
args = ap.parse_args()

PKG = {'metric': ('metric_baseline', ''), 'shape': ('shape_metric', '--arch original --rnl none'),
       'lie': ('lie_metric', '--arch original --rnl none')}
SUITES = {
    'Tshape3d_env1': 'Tshape3d_env1 Tshape3d_env1_tight_3d Tshape3d_env1_tight_3d_exp1 Tshape3d_env1_tight_3d_exp2 '
                     'Tshape3d_env1_tight_3d_exp3 Tshape3d_env1_tight001_3d Tshape3d_env1_rotheavy '
                     'Tshape3d_env1_bigangle Tshape3d_env1_transonly',
    'Lshape3d_env3': 'Lshape3d_env3 Lshape3d_env3_tight_3d Lshape3d_env3_tight_3d_exp1 Lshape3d_env3_tight_3d_exp2 Lshape3d_env3_tight_3d_exp3',
    'Ashape3d_env4': 'Ashape3d_env4 Ashape3d_env4_tight_3d Ashape3d_env4_tight_3d_exp1 Ashape3d_env4_tight_3d_exp2 Ashape3d_env4_tight_3d_exp3',
}
ORDER = [('Tshape3d_env1', k, 2) for k in ('lie', 'shape', 'metric')] + \
        [('Lshape3d_env3', k, 1) for k in ('lie', 'shape', 'metric')] + \
        [('Tshape3d_env1', k, 3) for k in ('lie', 'shape', 'metric')] + \
        [('Ashape3d_env4', k, 1) for k in ('lie', 'shape', 'metric')] + \
        [(c, k, sd) for sd in (2, 3) for c in ('Lshape3d_env3', 'Ashape3d_env4') for k in ('lie', 'shape', 'metric')]


def tag(cell, kind, seed):
    return '{}_s{}'.format(kind, seed) if cell == 'Tshape3d_env1' else '{}_{}_s{}'.format(cell, kind, seed)


def free_mb():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        i, u, t = (int(x) for x in line.split(','))
        res[i] = t - u
    return res


def job_script(cell, kind, seed):
    pkg, extra = PKG[kind]
    name = '{}_{}_s{}'.format(cell, kind, seed)
    t = tag(cell, kind, seed)
    ck_glob = 'Experiments/compare/{}/Model_Epoch_04000_*.pt'.format(name)
    return name, (
        'set -e; OMP_NUM_THREADS=4 python train/train_3dshape.py --dataPath datasets/3dshape/{cell}/ '
        '--models {pkg} {extra} --epochs 4000 --device cuda:0 --modelPath ./Experiments/compare '
        '--name {name} --no-wandb > Experiments/compare/{name}_train.log 2>&1; '
        'ck=$(ls {ck_glob}); '
        'bash analysis/run_plan_suite.sh {kind} $ck {t} {sets} > results/compare/plan/{t}.suite.log 2>&1'
    ).format(cell=cell, pkg=pkg, extra=extra, name=name, ck_glob=ck_glob, kind=kind, t=t, sets=SUITES[cell])


os.makedirs('Experiments/compare', exist_ok=True)
os.makedirs('results/compare/plan', exist_ok=True)
gpus = [int(g) for g in args.gpus.split(',')]
pending = [j for j in ORDER if not args.jobs or '{}_{}_s{}'.format(*j) in args.jobs.split(',')]
if args.jobs:  # keep the order given on the command line
    pending.sort(key=lambda j: args.jobs.split(',').index('{}_{}_s{}'.format(*j)))
running = []
while pending or running:
    running = [(n, p) for n, p in running if p.poll() is None]
    if pending and len(running) < args.max_jobs:
        fm = free_mb()
        need = args.need_metric if pending[0][1] == 'metric' else args.need
        cand = sorted((fm[g], g) for g in gpus if fm.get(g, 0) >= need)
        if cand:
            g = cand[-1][1]
            cell, kind, seed = pending.pop(0)
            name, cmd = job_script(cell, kind, seed)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            p = subprocess.Popen(['bash', '-c', cmd], env=env)
            running.append((name, p))
            print('{} start {} on GPU {} ({} MB free)'.format(time.strftime('%H:%M:%S'), name, g, fm[g]), flush=True)
            time.sleep(args.settle)
            continue
    time.sleep(30)
print('{} all jobs done'.format(time.strftime('%H:%M:%S')), flush=True)
