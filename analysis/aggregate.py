"""Aggregate the comparison: planning suites + field analyses -> summary.json / summary.md.

    python analysis/aggregate.py [--root results/compare]

Expects results/compare/plan/<kind>_s<seed>/<set>.json (analysis/plan_eval.py),
results/compare/field/<kind>_s<seed>.json (analysis/field_analysis.py) and
results/compare/linvalid/<set>.json (analysis/lin_valid.py).
"""
import os
import re
import glob
import json
import argparse
from math import comb

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='results/compare')
args = ap.parse_args()
R = args.root
KINDS = ['metric', 'shape', 'lie']
SETS_T = [('Tshape3d_env1', 'standard 1k (offset 0.02)'),
          ('Tshape3d_env1_tight_3d', 'tight (offset 0.005)'),
          ('Tshape3d_env1_tight001_3d', 'very tight (offset 0.001)'),
          ('Tshape3d_env1_tight_3d_exp1', 'non-linear (straight line collides)'),
          ('Tshape3d_env1_tight_3d_exp2', 'farthest (largest displacement)'),
          ('Tshape3d_env1_tight_3d_exp3', 'hard + far'),
          ('Tshape3d_env1_rotheavy', 'rotation-heavy (|dp|<=0.05, geo>=90 deg)'),
          ('Tshape3d_env1_bigangle', 'large angles (both >=135 deg)'),
          ('Tshape3d_env1_transonly', 'translation only (control)')]


def cell_sets(c):
    return [(c, c + ': standard 1k'), (c + '_tight_3d', c + ': tight (offset 0.005)'),
            (c + '_tight_3d_exp1', c + ': non-linear'),
            (c + '_tight_3d_exp2', c + ': farthest'), (c + '_tight_3d_exp3', c + ': hard + far')]


CELLS = {'Tshape3d_env1': ('', SETS_T),
         'Lshape3d_env3': ('Lshape3d_env3_', cell_sets('Lshape3d_env3')),
         'Ashape3d_env4': ('Ashape3d_env4_', cell_sets('Ashape3d_env4'))}
SETS = [x for c in CELLS for x in CELLS[c][1]]


def tags_of(kind, prefix=''):
    t = sorted({os.path.basename(p) for p in glob.glob(os.path.join(R, 'plan', prefix + kind + '_s*'))
               if os.path.isdir(p)})
    return t


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'n': len(v), 'values': v}


summary = {'sets': {}, 'field': {}, 'pairs': {}, 'tags': {}, 'cells': {c: [x[0] for x in CELLS[c][1]] for c in CELLS}}
for s, label, tags in [(x[0], x[1], {k: tags_of(k, CELLS[c][0]) for k in KINDS}) for c in CELLS for x in CELLS[c][1]]:
    summary['tags'][s] = tags
    lv = load(os.path.join(R, 'linvalid', s + '.json'))
    lvd = {r['case']: r for r in lv} if lv else {}
    entry = {'label': label, 'kinds': {}}
    recs_by_tag = {}
    for k in KINDS:
        rows = []
        for t in tags[k]:
            d = load(os.path.join(R, 'plan', t, s + '.json'))
            if d is None:
                continue
            recs_by_tag[t] = {r['case']: r for r in d['records']}
            sm = d['summary']
            rows.append(sm)
        if not rows:
            continue
        e = {key: ms([r.get(key) for r in rows]) for key in
             ('success_rate', 'collision_rate', 'nonconv_rate', 'median_iters_success',
              'median_trans_ratio_success', 'median_rotgeo_ratio_success',
              'median_rot_len_geo_success', 'median_trans_len_success',
              'median_min_clearance_success')}
        e['n_endpoints_ok'] = rows[0]['n_endpoints_ok']
        # tau as a cost-to-go: rank correlation of tau(start, goal) with the realised
        # path cost (translation + geodesic rotation) over successes, per seed; and
        # how often tau(waypoint, goal) increases along executed paths.
        from scipy.stats import spearmanr
        rho, inc = [], []
        for t in tags[k]:
            if t not in recs_by_tag:
                continue
            ok = [r for r in recs_by_tag[t].values() if r['success']]
            if len(ok) > 10:
                rho.append(float(spearmanr([r['tau_start'] for r in ok],
                                           [r['trans_len'] + r['rot_len_geo'] for r in ok]).correlation))
            inc.append(float(np.mean([r['tau_frac_increase'] for r in recs_by_tag[t].values()])))
        e['tau_vs_pathcost_spearman'] = ms(rho)
        e['mean_tau_frac_increase'] = ms(inc)
        # breakdowns (pooled over seeds)
        bd = {}
        if lvd:
            def rate(pred):
                v = [r['success'] for t in tags[k] if t in recs_by_tag
                     for c, r in recs_by_tag[t].items() if c in lvd and pred(lvd[c])]
                return [float(np.mean(v)) if v else None, len(v)]
            bd['lin_valid'] = rate(lambda x: x['lin_valid'])
            bd['lin_invalid'] = rate(lambda x: not x['lin_valid'])
            for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
                bd['rotgeo_{}_{}'.format(lo, hi)] = rate(lambda x, lo=lo, hi=hi: lo <= x['rot_geo_deg'] < hi)
            for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 2.0)):
                bd['trans_{}_{}'.format(lo, hi)] = rate(lambda x, lo=lo, hi=hi: lo <= x['trans_dist'] < hi)
            for lo, hi in ((0, 90), (90, 135), (135, 181)):
                bd['maxtheta_{}_{}'.format(lo, hi)] = rate(
                    lambda x, lo=lo, hi=hi: lo <= max(x['theta_start_deg'], x['theta_goal_deg']) < hi)
        e['breakdown'] = bd
        entry['kinds'][k] = e
    # paired comparisons, seed-matched (s1 vs s1, ...)
    pairs = {}
    for a, b in (('lie', 'shape'), ('lie', 'metric'), ('shape', 'metric')):
        tot = {'both': 0, 'only_' + a: 0, 'only_' + b: 0, 'neither': 0}
        eff = {'rot_geo_ratio': [], 'trans_ratio': [], 'iters_ratio': [], 'clear_diff': []}
        per_seed = []
        for ta, tb in zip(tags[a], tags[b]):
            if ta not in recs_by_tag or tb not in recs_by_tag:
                continue
            A, B = recs_by_tag[ta], recs_by_tag[tb]
            common = sorted(set(A) & set(B))
            n11 = sum(A[c]['success'] and B[c]['success'] for c in common)
            n10 = sum(A[c]['success'] and not B[c]['success'] for c in common)
            n01 = sum((not A[c]['success']) and B[c]['success'] for c in common)
            n00 = len(common) - n11 - n10 - n01
            per_seed.append({'seeds': [ta, tb], 'both': n11, 'only_' + a: n10, 'only_' + b: n01,
                             'neither': n00, 'p_mcnemar': mcnemar(n10, n01)})
            tot['both'] += n11; tot['only_' + a] += n10; tot['only_' + b] += n01; tot['neither'] += n00
            for c in common:
                if A[c]['success'] and B[c]['success']:
                    ra, rb = A[c], B[c]
                    if rb['rot_len_geo'] > 1e-3:
                        eff['rot_geo_ratio'].append(ra['rot_len_geo'] / rb['rot_len_geo'])
                    if rb['trans_len'] > 1e-3:
                        eff['trans_ratio'].append(ra['trans_len'] / rb['trans_len'])
                    if ra['iters'] and rb['iters']:
                        eff['iters_ratio'].append((ra['iters'] + 1) / (rb['iters'] + 1))
                    eff['clear_diff'].append(ra['min_clearance'] - rb['min_clearance'])
        if per_seed:
            tot['p_mcnemar_pooled'] = mcnemar(tot['only_' + a], tot['only_' + b])
            tot['per_seed'] = per_seed
            tot['common_success_median'] = {k2: (float(np.median(v)) if v else None) for k2, v in eff.items()}
            tot['common_success_n'] = len(eff['clear_diff'])
            pairs['{}_vs_{}'.format(a, b)] = tot
    entry['pairs'] = pairs
    summary['sets'][s] = entry

# ------------------------------------------------------------------ RNL suite (two-route network)
# tags: rnl_<eikonal>_<probe> (seed 1) and rnl_<eikonal>_<probe>_s<k> (further seeds)
RNL_SETS = ['Tshape3d_env1', 'Tshape3d_env1_tight_3d', 'Tshape3d_env1_tight_3d_exp1', 'Tshape3d_env1_tight_3d_exp2',
            'Tshape3d_env1_rotheavy', 'Tshape3d_env1_bigangle']
rs = {}
for sid in RNL_SETS:
    e = {'label': dict(SETS).get(sid, sid), 'rates': {}, 'pairs': {}}
    recs = {}
    for eik in ('shape', 'lie'):
        for probe in ('none', 'orig', 'lie'):
            base = 'rnl_{}_{}'.format(eik, probe)
            for seed, t in ((1, base), (2, base + '_s2'), (3, base + '_s3')):
                d = load(os.path.join(R, 'plan', t, sid + '.json'))
                if d:
                    recs[(eik, probe, seed)] = {r['case']: r['success'] for r in d['records']}
                    e['rates'].setdefault('rnl_{}_{}'.format(eik, probe), {}).setdefault('seeds', {})[seed] = d['summary']['success_rate']
                    e['rates']['rnl_{}_{}'.format(eik, probe)]['n'] = d['summary']['n_endpoints_ok']
    for t, v in e['rates'].items():
        v['success'] = float(np.mean(list(v['seeds'].values())))
    for eik in ('shape', 'lie'):
        for a_, b_ in (('lie', 'orig'), ('orig', 'none'), ('lie', 'none')):
            n10 = n01 = 0
            used = []
            for seed in (1, 2, 3):
                A, B = recs.get((eik, a_, seed)), recs.get((eik, b_, seed))
                if A is None or B is None:
                    continue
                c = sorted(set(A) & set(B))
                n10 += sum(A[i] and not B[i] for i in c); n01 += sum(B[i] and not A[i] for i in c)
                used.append(seed)
            if used:
                e['pairs']['{}: {} vs {}'.format(eik, a_, b_)] = {'only_A': n10, 'only_B': n01, 'p': mcnemar(n10, n01), 'seeds': used}
    if e['rates']:
        rs[sid] = e
summary['rnl_suite'] = rs

# ------------------------------------------------------------------ field analyses
for k in KINDS:
    fs = [load(p) for p in sorted(glob.glob(os.path.join(R, 'field', k + '_s*.json')))]
    fs = [f for f in fs if f]
    if not fs:
        continue
    summary['field'][k] = {'n_seeds': len(fs), 'tags': [f['tag'] for f in fs], 'runs': fs}

# field analyses in the other environments: field/<cell>_<kind>_s<seed>.json
summary['field_cells'] = {}
for c in CELLS:
    if c == 'Tshape3d_env1':
        continue
    for k in KINDS:
        fs = [load(p) for p in sorted(glob.glob(os.path.join(R, 'field', '{}_{}_s*.json'.format(c, k))))]
        fs = [f for f in fs if f]
        if fs:
            summary['field_cells'].setdefault(c, {})[k] = {'n_seeds': len(fs), 'runs': fs}

json.dump(summary, open(os.path.join(R, 'summary.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------ markdown
L = ['# metric vs shape vs lie NTFields (Tshape3d_env1, metric architecture, Eikonal only)', '']
L.append('Seeds per cell: ' + '; '.join('{}: '.format(c) + ', '.join('{}={}'.format(k, len(tags_of(k, CELLS[c][0]))) for k in KINDS) for c in CELLS))
L.append('')
L.append('## Planning success (forward MPPI), mean +- std over seeds')
L.append('| test set | n | metric | shape | lie |')
L.append('|---|---|---|---|---|')


def fmt(e, key='success_rate', pct=True):
    if not e or not e.get(key):
        return '--'
    v = e[key]
    if pct:
        return '{:.1f} +- {:.1f}'.format(100 * v['mean'], 100 * v['std'])
    return '{:.3f} +- {:.3f}'.format(v['mean'], v['std'])


for s, label in SETS:
    e = summary['sets'].get(s, {}).get('kinds', {})
    if not e:
        continue
    n = next(iter(e.values()))['n_endpoints_ok']
    L.append('| {} | {} | {} | {} | {} |'.format(label, n, *(fmt(e.get(k)) for k in KINDS)))
L.append('')
for key, title in (('collision_rate', 'Collision rate'), ('nonconv_rate', 'Non-convergence rate'),
                   ('median_rotgeo_ratio_success', 'Path rotation (geodesic) / straight geodesic, successes'),
                   ('median_trans_ratio_success', 'Path translation / straight translation, successes'),
                   ('median_min_clearance_success', 'Min translational clearance along path, successes'),
                   ('tau_vs_pathcost_spearman', 'Spearman(tau(start,goal), realised path cost), successes'),
                   ('mean_tau_frac_increase', 'Mean fraction of path steps where tau(waypoint, goal) increases')):
    L.append('## ' + title)
    L.append('| test set | metric | shape | lie |')
    L.append('|---|---|---|---|')
    for s, label in SETS:
        e = summary['sets'].get(s, {}).get('kinds', {})
        if not e:
            continue
        pct = key.endswith('_rate')
        L.append('| {} | {} | {} | {} |'.format(label, *(fmt(e.get(k), key, pct) for k in KINDS)))
    L.append('')
L.append('## Paired per-case comparisons (seed-matched, pooled over seeds)')
L.append('| test set | pair | both | only A | only B | neither | McNemar p | A/B rot-geo length (both solved) | A/B translation length |')
L.append('|---|---|---|---|---|---|---|---|---|')
for s, label in SETS:
    P = summary['sets'].get(s, {}).get('pairs', {})
    for pk, v in P.items():
        a, b = pk.split('_vs_')
        cm = v['common_success_median']
        L.append('| {} | {} | {} | {} | {} | {} | {:.2g} | {} | {} |'.format(
            label, pk.replace('_', ' '), v['both'], v['only_' + a], v['only_' + b], v['neither'],
            v['p_mcnemar_pooled'],
            '{:.3f}'.format(cm['rot_geo_ratio']) if cm['rot_geo_ratio'] else '--',
            '{:.3f}'.format(cm['trans_ratio']) if cm['trans_ratio'] else '--'))
L.append('')
# ------------------------------------------------------------------ field tables
F = summary['field']


def fget(k, path):
    vals = []
    for run in F.get(k, {}).get('runs', []):
        v = run
        try:
            for p in path:
                v = v[p]
        except (KeyError, TypeError):
            v = None
        if v is not None:
            vals.append(v)
    return ms(vals)


def ff(e, nd=3):
    if not e:
        return '--'
    return ('{:.%df} +- {:.%df}' % (nd, nd)).format(e['mean'], e['std'])


if F:
    L.append('## Field: local metric identification (probe points, h = 0.05)')
    L.append('| statistic | metric | shape | lie |')
    L.append('|---|---|---|---|')
    for lab, path in (('norm exponent p (median); 1 = times add, 2 = quadrature', ['local_metric', 'p_exponent', 'median']),
                      ('fraction of points with p < 1.5', ['local_metric', 'p_exponent', 'frac_p_lt_1p5']),
                      ('sqrt2 tau(45) / (tau(0) + tau(90))', ['local_metric', 'additivity_ratio_median']),
                      ('free pairs R2: Euclidean, chart', ['free_metric', 'euclid_chart', 'R2']),
                      ('free pairs R2: block-L1, chart', ['free_metric', 'L1_chart', 'R2']),
                      ('free pairs R2: Euclidean, SO(3) geodesic', ['free_metric', 'euclid_geodesic', 'R2']),
                      ('free pairs R2: block-L1, SO(3) geodesic', ['free_metric', 'L1_geodesic', 'R2'])):
        L.append('| {} | {} | {} | {} |'.format(lab, *(ff(fget(k, path)) for k in KINDS)))
    L.append('')
    L.append('## Field: dual norms of the start gradient vs goal direction phi (0 = pure translation, 90 = pure rotation)')
    L.append('| phi | quantity | metric | shape | lie |')
    L.append('|---|---|---|---|---|')
    for ph in ('0.0', '22.5', '45.0', '67.5', '90.0'):
        for qn in ('x=Sd|g_p|', 'y=Sa|g_w|', 'x2+y2', 'iso=Sc|g|'):
            L.append('| {} | {} | {} | {} | {} |'.format(ph, qn, *(ff(fget(k, ['local_metric', 'per_phi', ph, qn])) for k in KINDS)))
    L.append('')
    L.append('## Field: speed calibration (median |log(pred/target)|; 0 = exact)')
    L.append('| speed | metric | shape | lie |')
    L.append('|---|---|---|---|')
    for key, lab in (('dir_t', 'directional translational vs S*_d'), ('dir_r', 'directional rotational (body) vs S*_a'),
                     ('t', 'gradient 1/|g_p| vs S*_d'), ('rc', 'gradient 1/|g_r| (chart) vs S*_a'),
                     ('rw', 'gradient 1/|g_w| (body) vs S*_a'), ('iso', 'gradient 1/|g| vs S*_c (baseline target)')):
        L.append('| {} | {} | {} | {} |'.format(lab, *(ff(fget(k, ['global', 'median_abslog', key])) for k in KINDS)))
    for key, lab in (('Sdir_t~Sstar_d', 'Spearman directional translational vs S*_d'),
                     ('Sdir_r~Sstar_a', 'Spearman directional rotational vs S*_a')):
        L.append('| {} | {} | {} | {} |'.format(lab, *(ff(fget(k, ['global', 'spearman', key])) for k in KINDS)))
    for key, lab in (('aniso', 'own-or-not residual |N_aniso - 1|'), ('blk_t', '|S*_d|g_p| - 1|'), ('blk_r', '|S*_a|g_r| - 1|'),
                     ('iso', '|S*_c|g| - 1|')):
        L.append('| {} | {} | {} | {} |'.format(lab, *(ff(fget(k, ['global', 'median_absres', key])) for k in KINDS)))
    for key, lab in (('Sdir_t', 'dynamic range free/tight, directional translational'),
                     ('Sdir_r', 'dynamic range free/tight, directional rotational'),
                     ('Sstar_d', 'dynamic range of the target S*_d'), ('Sstar_a', 'dynamic range of the target S*_a')):
        L.append('| {} | {} | {} | {} |'.format(lab, *(ff(fget(k, ['global', 'dynamic_range', key]), 2) for k in KINDS)))
    L.append('')
    L.append('## Field: goal dependence of the speed at a fixed point (median CV over 16 goals)')
    L.append('| speed | metric | shape | lie |')
    L.append('|---|---|---|---|')
    for key in ('S_t', 'S_rc', 'S_rw', 'S_iso'):
        L.append('| {} | {} | {} | {} |'.format(key, *(ff(fget(k, ['goal_cv', key])) for k in KINDS)))
    L.append('')
    L.append('## Field: rotation tests')
    L.append('| statistic | metric | shape | lie |')
    L.append('|---|---|---|---|')
    for lab, path in (('tau(wrap) / tau(control), paired median (1 = SO(3)-aware)', ['rot_pairs', 'paired_wrap_over_control_median']),
                      ('chart distance ratio wrap/control', ['rot_pairs', 'chart_ratio_wrap_over_control_median']),
                      ('pure rotations: R2 of tau vs geodesic angle', ['rot_pairs', 'generic_fit_geodesic', 'R2']),
                      ('pure rotations: R2 of tau vs chart distance', ['rot_pairs', 'generic_fit_chart', 'R2']),
                      ('cut locus: median |dtau|/tau, same pose two rotvecs', ['cut_locus', 'median_rel_tau_diff']),
                      ('cut locus: p90 |dtau|/tau', ['cut_locus', 'p90_rel_tau_diff']),
                      ('cut locus: |log speed ratio|, body rotational', ['cut_locus', 'median_log_speed_ratio', 'S_rw'])):
        L.append('| {} | {} | {} | {} |'.format(lab, *(ff(fget(k, path)) for k in KINDS)))
    L.append('')
    L.append('## Field: band probes (fraction of band points where the speed drops along the clearance-reducing step)')
    L.append('| probe / speed | metric | shape | lie |')
    L.append('|---|---|---|---|')
    for mode in ('orig', 'lie'):
        for grp, sp in (('trans_probe_frac_drop', 'S_t'), ('trans_probe_frac_drop', 'S_iso'),
                        ('rot_probe_frac_drop', 'S_rc'), ('rot_probe_frac_drop', 'S_rw'), ('rot_probe_frac_drop', 'S_iso')):
            L.append('| {} {} {} | {} | {} | {} |'.format(mode, grp.split('_')[0], sp,
                                                          *(ff(fget(k, ['band_probe', mode, grp, sp])) for k in KINDS)))
    L.append('')
    # binned speed tables: directional speeds vs targets per (dist, angle) bin
    E = None
    for k in KINDS:
        if F.get(k):
            E = F[k]['runs'][0]['binned']['edges']
            break
    if E:
        de, ae = E['dist_over_margin'], E['angle_deg']
        dl = ['{}-{}'.format(de[i], de[i + 1]) for i in range(len(de) - 1)]
        al = ['{}-{}'.format(ae[j], ae[j + 1]) for j in range(len(ae) - 1)]
        for key, lab in (('Sdir_t', 'directional translational speed'), ('Sdir_r', 'directional rotational speed'),
                         ('Sstar_d', 'target S*_d'), ('Sstar_a', 'target S*_a')):
            for k in KINDS:
                if not F.get(k):
                    continue
                if key.startswith('Sstar') and k != 'metric':
                    continue
                L.append('### {} -- {} (rows: d/margin; cols: rotational clearance deg; median over seeds of bin medians)'.format(
                    lab, k if not key.startswith('Sstar') else 'geometry'))
                L.append('| d/margin | ' + ' | '.join(al) + ' |')
                L.append('|---' * (len(al) + 1) + '|')
                for i in range(len(dl)):
                    cells = []
                    for j in range(len(al)):
                        e = fget(k, ['binned', 'grid', '{}_{}'.format(i, j), key])
                        n = fget(k, ['binned', 'grid', '{}_{}'.format(i, j), 'n'])
                        cells.append('--' if not e or not n or n['mean'] < 20 else '{:.2f}'.format(e['mean']))
                    L.append('| {} | '.format(dl[i]) + ' | '.join(cells) + ' |')
                L.append('')
open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L))
print('\n'.join(L))
