"""Compare the original and Lie MPPI against a historical plan_eval log.

    python analysis/compare_lie_mppi.py \
        --historical results/compare/plan/lie_rsqrtN_s1_g/Tshape3d_env1.json \
        --out results/lie_mppi_comparison/my_run --device cuda:0

Both new runs use the same checkpoint, cases, cloud seed, batch and budget.
The output directory must be new; historical evidence is never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def counts(result, indices=None):
    records = result['records']
    if indices is not None:
        records = [r for r in records if r['case'] in indices]
    return {'cases': len(records),
            'success': sum(r['success'] for r in records),
            'collision': sum(r['collision'] for r in records),
            'nonconverged': sum(not r['converged'] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch', type=int, default=125)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--cases', type=int, default=0, help='0 = all frozen cases')
    args = parser.parse_args()
    historical_path = Path(args.historical)
    historical = json.loads(historical_path.read_text())
    reference = historical['summary']
    if not reference.get('group_conv') or reference.get('planner', 'mppi') != 'mppi':
        parser.error('Use an original MPPI log with group_conv=true for matched convergence.')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    results = {}
    for name, planner in [('original', 'mppi'), ('lie', 'mppi_lie')]:
        command = [sys.executable, '-u', 'analysis/plan_eval.py',
                   '--kind', reference['kind'], '--checkpoint', reference['checkpoint'],
                   '--dataPath', reference['dataPath'], '--out', str(output / (name + '.json')),
                   '--device', args.device, '--seed', str(args.seed), '--batch', str(args.batch),
                   '--workers', str(args.workers), '--cases', str(args.cases),
                   '--steps', '200', '--temp', '50', '--group-conv', '--planner', planner]
        commands.append(command)
        print('Running ' + name + ': ' + ' '.join(command), flush=True)
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[name] = json.loads((output / (name + '.json')).read_text())
        print(name + ': ' + json.dumps(counts(results[name])), flush=True)

    original = {r['case']: r for r in results['original']['records']}
    lie = {r['case']: r for r in results['lie']['records']}
    if original.keys() != lie.keys():
        raise ValueError('Endpoint filtering differs between the two new runs.')
    paired = {
        'cases': len(original),
        'rescued': sum(lie[i]['success'] and not original[i]['success'] for i in original),
        'lost': sum(original[i]['success'] and not lie[i]['success'] for i in original),
        'both_success': sum(original[i]['success'] and lie[i]['success'] for i in original),
        'both_failure': sum(not original[i]['success'] and not lie[i]['success'] for i in original),
    }
    files = [reference['checkpoint'], historical_path,
             Path(reference['dataPath']) / 'sampled_points.npy',
             Path(reference['dataPath']) / 'meta.json',
             'controllers/lie_mppi.py', 'analysis/plan_eval.py',
             'analysis/compare_lie_mppi.py', 'models/rnl.py']
    import numpy as np
    import torch
    evidence = {
        'historical': str(historical_path), 'commands': commands, 'paired': paired,
        'historical_counts': counts(historical),
        'historical_counts_on_current_cases': counts(historical, set(original)),
        'original_counts': counts(results['original']), 'lie_counts': counts(results['lie']),
        'environment': {'python': sys.version, 'torch': torch.__version__,
                        'numpy': np.__version__, 'device': args.device},
        'sha256': {str(p): sha256(p) for p in files},
    }
    (output / 'comparison.json').write_text(json.dumps(evidence, indent=2) + '\n')
    report = [
        '# Lie MPPI comparison', '',
        f"Checkpoint: `{reference['checkpoint']}`.",
        f"Frozen cases: `{reference['dataPath']}`; planning/cloud seed: {args.seed}.", '',
        'The new controller uses unit body tangents and the Riemannian exponential',
        'for d² = ||Δp||² + (relative rotation angle / 2π)². Each candidate step',
        'has common t = 0.015 and geodesic length 0.015. The weighted mean is',
        'executed in the current tangent space with the same exponential.', '',
        'Both fresh runs use 50 candidates, horizon 5, 200 iterations, temperature 50,',
        'zero momentum and product-geodesic convergence at 0.01. Costs and mesh',
        'collision checks are unchanged. The historical log also uses group convergence.', '',
        '| Run | Valid cases | Success | Collision | No convergence |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    rows = [('Historical original', counts(historical)),
            ('Historical original, matching case indices', counts(historical, set(original))),
            ('Fresh original', counts(results['original'])), ('Lie MPPI', counts(results['lie']))]
    for label, row in rows:
        n = row['cases']
        cells = [f"{row[key]}/{n} ({100 * row[key] / n:.1f}%)" if n else 'n/a'
                 for key in ['success', 'collision', 'nonconverged']]
        report.append('| ' + ' | '.join([label, str(n)] + cells) + ' |')
    report += ['', f"Paired outcomes: {paired['rescued']} rescued, {paired['lost']} lost, "
               f"{paired['both_success']} both succeed, {paired['both_failure']} both fail.", '',
               'Collision and nonconvergence overlap. Success requires convergence and',
               'no collision at any recorded waypoint, including the appended goal.',
               'This uses the existing waypoint collision protocol; continuous segment',
               'collision safety is not established by that protocol.', '',
               'Wall times below compare the two fresh runs only; historical timings',
               'use a different machine/environment. This is one checkpoint and planning',
               'seed, not a multi-seed statistical generalization study.', '']
    for name in ['original', 'lie']:
        seconds = results[name]['summary']['seconds']
        report.append(f"- {name}: planning {seconds['plan_gpu']:.2f} s; "
                      f"collision checks {seconds['collision_cpu']:.2f} s; device {args.device}.")
    report += ['', 'Exact commands, versions, input/source hashes and per-case results',
               'are in `comparison.json`, `original.json`, `lie.json` and the corresponding logs.', '']
    (output / 'REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
