"""Plan-eval success split by the start -> goal relative rotation (geodesic angle).

Reads results/compare/plan/<tag>/<set>.json records written by analysis/plan_eval.py
and pools repeats of a group.  Each group is `name=tag1,tag2,...` (tags may be
globs); the set defaults to Tshape3d_env1.

    python3 analysis/plan_by_angle.py \
        'chart=lie_rsqrtN_s[123]_g' 'rot6f1=lie_rsqrtN_rot6f1_s[123]_g' 'sheets=lie_rsqrtN_quatsheets_s[123]_g'

Per bin: s = success %, nc = not converged %, c = collided % (a case can be both).
Needs only numpy (runs on the host python).
"""
import os
import sys
import glob
import json
import argparse

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('groups', nargs='+', metavar='NAME=TAGGLOB[,TAGGLOB]')
ap.add_argument('--set', default='Tshape3d_env1')
ap.add_argument('--root', default='results/compare/plan')
ap.add_argument('--bins', default='0,30,60,90,120,150,165,181')
args = ap.parse_args()
bins = [int(b) for b in args.bins.split(',')]

print('{} ({}): success by relative rotation, pooled over repeats'.format(args.set, args.root))
for spec in args.groups:
    name, globs = spec.split('=', 1)
    recs, tags = [], []
    for g in globs.split(','):
        for d in sorted(glob.glob(os.path.join(args.root, g))):
            p = os.path.join(d, args.set + '.json')
            if os.path.exists(p):
                recs += json.load(open(p))['records']
                tags.append(os.path.basename(d))
    if not recs:
        print('  {:<12} no records'.format(name))
        continue
    geo = np.array([r['straight_rot_geo'] * 360 for r in recs])
    succ = np.array([r['success'] for r in recs])
    conv = np.array([r['converged'] for r in recs])
    coll = np.array([r['collision'] for r in recs])
    line = '  {:<12} {} runs, n={:<5d} all: s{:4.1f} nc{:4.1f} c{:4.1f} |'.format(
        name, len(tags), len(recs), 100 * succ.mean(), 100 * (~conv).mean(), 100 * coll.mean())
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (geo >= lo) & (geo < hi)
        if m.any():
            line += ' {}-{}: s{:3.0f} nc{:3.0f} c{:3.0f} (n{}) |'.format(
                lo, min(hi, 180), 100 * succ[m].mean(), 100 * (~conv[m]).mean(), 100 * coll[m].mean(), m.sum())
    print(line)
