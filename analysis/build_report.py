"""Build the comparison report page from results/compare/summary.json.

    python analysis/build_report.py --out results/compare/report.html

Also reads results/compare/findings.json (list of HTML strings, written by hand
after looking at the numbers) and the relative-normal-loss evaluations under
results/output_3d/*_sntf_ep0*/success_rate.txt.
"""
import os
import re
import json
import time
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='results/compare')
ap.add_argument('--out', default='results/compare/report.html')
ap.add_argument('--rnl_diag', default='')
args = ap.parse_args()
R = args.root
S = json.load(open(os.path.join(R, 'summary.json')))
KINDS = ['metric', 'shape', 'lie']


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'n': len(vals), 'values': [float(v) for v in vals]}


# ------------------------------------------------------------------ planning sets
sets = []
for sid, e in S['sets'].items():
    if not e.get('kinds'):
        continue
    cell = next((c for c, ids in S['cells'].items() if sid in ids), 'Tshape3d_env1')
    ks = {}
    for k, v in e['kinds'].items():
        ks[k] = {'success': v.get('success_rate'), 'collision': v.get('collision_rate'),
                 'nonconv': v.get('nonconv_rate'), 'rotgeo': v.get('median_rotgeo_ratio_success'),
                 'transr': v.get('median_trans_ratio_success'), 'clear': v.get('median_min_clearance_success'),
                 'tauspear': v.get('tau_vs_pathcost_spearman'), 'tauinc': v.get('mean_tau_frac_increase')}
    n = next(iter(e['kinds'].values()))['n_endpoints_ok']
    sets.append({'id': sid, 'label': e['label'], 'cell': cell, 'n': n, 'kinds': ks, 'breakdown': {k: v.get('breakdown', {}) for k, v in e['kinds'].items()}})
order = [x for c in S['cells'] for x in S['cells'][c]]
sets.sort(key=lambda s: order.index(s['id']) if s['id'] in order else 999)


def pooled(prefixes, tshape_only=True):
    labels, series, counts = [], {k: [] for k in KINDS}, {k: [] for k in KINDS}
    for key, lab in prefixes:
        labels.append(lab)
        for k in KINDS:
            num = den = 0
            for s in sets:
                if tshape_only and s['cell'] != 'Tshape3d_env1':
                    continue
                b = s['breakdown'].get(k, {}).get(key)
                if b and b[0] is not None and b[1]:
                    num += b[0] * b[1]; den += b[1]
            series[k].append(num / den if den else None)
            counts[k].append(den)
    return {'labels': labels, 'series': series, 'counts': counts}


breakdown = {
    'rotgeo': pooled([('rotgeo_0_45', '0–45°'), ('rotgeo_45_90', '45–90°'), ('rotgeo_90_135', '90–135°'), ('rotgeo_135_181', '135–180°')]),
    'trans': pooled([('trans_0_0.2', '0–0.2'), ('trans_0.2_0.4', '0.2–0.4'), ('trans_0.4_0.6', '0.4–0.6'), ('trans_0.6_2.0', '> 0.6')]),
    'lin': pooled([('lin_valid', 'straight line free'), ('lin_invalid', 'straight line collides')]),
    'maxtheta': pooled([('maxtheta_0_90', '< 90°'), ('maxtheta_90_135', '90–135°'), ('maxtheta_135_181', '135–180°')]),
}

pairs = []
for s in sets:
    if s['cell'] != 'Tshape3d_env1':
        continue
    for pk, v in S['sets'][s['id']].get('pairs', {}).items():
        if pk == 'shape_vs_metric':
            continue
        a, b = pk.split('_vs_')
        cm = v['common_success_median']
        pairs.append({'label': s['label'], 'pair': '{} vs {}'.format(a, b), 'both': v['both'], 'onlyA': v['only_' + a],
                      'onlyB': v['only_' + b], 'neither': v['neither'], 'p': v['p_mcnemar_pooled'],
                      'rot': cm.get('rot_geo_ratio'), 'trans': cm.get('trans_ratio')})

# ------------------------------------------------------------------ field
F = S.get('field', {})


def runs(k):
    return F.get(k, {}).get('runs', [])


def path(d, p):
    for x in p:
        if d is None:
            return None
        d = d.get(x) if isinstance(d, dict) else None
    return d


def fms(k, p):
    return ms([path(r, p) for r in runs(k)])


edges = None
for k in KINDS:
    if runs(k):
        edges = runs(k)[0]['binned']['edges']
        break
field = {}
if edges:
    de, ae, te = edges['dist_over_margin'], edges['angle_deg'], edges['theta_deg']
    nd, na = len(de) - 1, len(ae) - 1

    def lab(lo, hi, fmt):
        return '≥ {}'.format(fmt(lo)) if hi in (float('inf'), None) or (isinstance(hi, float) and np.isinf(hi)) or hi > 180 else '{}–{}'.format(fmt(lo), fmt(hi))
    field['dLabels'] = [lab(de[i], de[i + 1], lambda v: ('%g' % v)) for i in range(nd)]
    field['aLabels'] = [lab(ae[j], ae[j + 1], lambda v: ('%g' % v)) for j in range(na)]
    field['thetaLabels'] = ['{}–{}°'.format(int(te[t]), int(round(te[t + 1]))) for t in range(len(te) - 1)]

    def grid(k, key, minn=20):
        g = []
        for i in range(nd):
            row = []
            for j in range(na):
                vals = [path(r, ['binned', 'grid', '{}_{}'.format(i, j), key]) for r in runs(k)]
                ns = [path(r, ['binned', 'grid', '{}_{}'.format(i, j), 'n']) or 0 for r in runs(k)]
                vals = [v for v, n in zip(vals, ns) if v is not None and n >= minn]
                row.append(float(np.mean(vals)) if vals else None)
            g.append(row)
        return g
    field['grid'] = {k: {key: grid(k, key) for key in ('Sdir_t', 'Sdir_r', 'logratio_dir_t', 'logratio_dir_r', 'S_t', 'S_rw', 'S_iso')}
                     for k in KINDS if runs(k)}
    k0 = next(k for k in KINDS if runs(k))
    field['target'] = {'Sstar_d': grid(k0, 'Sstar_d'), 'Sstar_a': grid(k0, 'Sstar_a')}
    field['count'] = [[path(runs(k0)[0], ['binned', 'grid', '{}_{}'.format(i, j), 'n']) or 0 for j in range(na)] for i in range(nd)]

    def marg(k, axis, key, n):
        out = []
        for i in range(n):
            v = [path(r, ['binned', axis, str(i), key]) for r in runs(k)]
            v = [x for x in v if x is not None]
            out.append(float(np.mean(v)) if v else None)
        return out
    field['marg'] = {'dist_t': {k: marg(k, 'dist', 'Sdir_t', nd) for k in KINDS if runs(k)},
                     'dist_target': marg(k0, 'dist', 'Sstar_d', nd),
                     'angle_r': {k: marg(k, 'angle', 'Sdir_r', na) for k in KINDS if runs(k)},
                     'angle_target': marg(k0, 'angle', 'Sstar_a', na)}
    field['theta'] = {k: marg(k, 'theta', 'logratio_dir_r', len(te) - 1) for k in KINDS if runs(k)}


def row(label, p, fmt='f2', higher=None):
    return {'label': label, 'vals': {k: fms(k, p) for k in KINDS}, 'fmt': fmt, 'higher': higher}


calib = [
    row('Directional translational speed vs S*d: median |log ratio| (lower is better)', ['global', 'median_abslog', 'dir_t'], higher=False),
    row('Directional rotational speed vs S*a: median |log ratio|', ['global', 'median_abslog', 'dir_r'], higher=False),
    row('Spearman, directional translational speed vs S*d', ['global', 'spearman', 'Sdir_t~Sstar_d'], higher=True),
    row('Spearman, directional rotational speed vs S*a', ['global', 'spearman', 'Sdir_r~Sstar_a'], higher=True),
    row('Speed ratio free / tight, translational (target 17)', ['global', 'dynamic_range', 'Sdir_t'], higher=True),
    row('Speed ratio free / tight, rotational (target 43)', ['global', 'dynamic_range', 'Sdir_r'], higher=True),
    row('Isotropic-equation residual |S*c·|∇τ| − 1|', ['global', 'median_absres', 'iso'], 'f3', False),
    row('Blocked-equation residual, translation |S*d·|∇pτ| − 1|', ['global', 'median_absres', 'blk_t'], 'f3', False),
    row('Blocked-equation residual, rotation |S*a·|∇rτ| − 1|', ['global', 'median_absres', 'blk_r'], 'f3', False),
    row('Anisotropic-equation residual |N − 1|', ['global', 'median_absres', 'aniso'], 'f3', False),
    row('Goal dependence of the translational speed at a fixed point (CV over 16 goals)', ['goal_cv', 'S_t'], 'f3', False),
    row('Goal dependence of the rotational speed (body) at a fixed point', ['goal_cv', 'S_rw'], 'f3', False),
    row('Band: speed drops along the clearance normal (translation probe)', ['band_probe', 'lie', 'trans_probe_frac_drop', 'S_t'], 'pct', True),
    row('Band: rotational speed drops along the Lie rotation probe', ['band_probe', 'lie', 'rot_probe_frac_drop', 'S_rw'], 'pct', True),
]
metric_rows = [
    row('Local norm exponent p, median (1 = costs add, 2 = quadrature)', ['local_metric', 'p_exponent', 'median']),
    row('Share of points with p < 1.5', ['local_metric', 'p_exponent', 'frac_p_lt_1p5'], 'pct'),
    row('√2·τ(45°) / (τ(0°) + τ(90°))  (1 = additive, 0.71 = quadrature, equal costs)', ['local_metric', 'additivity_ratio_median'], 'f3'),
    row('Free-space pairs, R²: Euclidean in the rotation-vector chart', ['free_metric', 'euclid_chart', 'R2'], 'f3'),
    row('Free-space pairs, R²: block-L1 in the chart', ['free_metric', 'L1_chart', 'R2'], 'f3'),
    row('Free-space pairs, R²: Euclidean with SO(3) geodesic rotation', ['free_metric', 'euclid_geodesic', 'R2'], 'f3'),
    row('Free-space pairs, R²: block-L1 with SO(3) geodesic rotation', ['free_metric', 'L1_geodesic', 'R2'], 'f3'),
]
rot_rows = [
    row('τ(wrap) / τ(control), paired median (1 = sees SO(3), large = chart-bound)', ['rot_pairs', 'paired_wrap_over_control_median']),
    row('Chart-distance ratio wrap / control (for reference)', ['rot_pairs', 'chart_ratio_wrap_over_control_median']),
    row('Pure rotations: R² of τ vs geodesic angle', ['rot_pairs', 'generic_fit_geodesic', 'R2'], 'f3'),
    row('Pure rotations: R² of τ vs chart distance', ['rot_pairs', 'generic_fit_chart', 'R2'], 'f3'),
    row('Cut locus: median |Δτ| / τ for one pose written two ways', ['cut_locus', 'median_rel_tau_diff'], 'f3'),
    row('Cut locus: 90th percentile |Δτ| / τ', ['cut_locus', 'p90_rel_tau_diff'], 'f3'),
    row('Cut locus: |log| ratio of body rotational speeds', ['cut_locus', 'median_log_speed_ratio', 'S_rw'], 'f3'),
    row('Band, chart rotation probe: rotational speed drops', ['band_probe', 'orig', 'rot_probe_frac_drop', 'S_rc'], 'pct'),
    row('Band, Lie rotation probe: rotational speed drops', ['band_probe', 'lie', 'rot_probe_frac_drop', 'S_rw'], 'pct'),
]
dual = {}
for k in KINDS:
    if not runs(k):
        continue
    pts = []
    for ph in ('0.0', '22.5', '45.0', '67.5', '90.0'):
        x = fms(k, ['local_metric', 'per_phi', ph, 'x=Sd|g_p|']); y = fms(k, ['local_metric', 'per_phi', ph, 'y=Sa|g_w|'])
        if x and y:
            pts.append({'phi': float(ph), 'x': x['mean'], 'y': y['mean']})
    dual[k] = pts

# ------------------------------------------------------------------ field in other environments
FC = S.get('field_cells', {})
cell_field_rows = []
for c, byk in FC.items():
    def cf(k, p):
        return ms([path(r, p) for r in byk.get(k, {}).get('runs', [])])
    for lab, p, fmt in (('local norm exponent p', ['local_metric', 'p_exponent', 'median'], 'f2'),
                        ('directional translational speed |log ratio|', ['global', 'median_abslog', 'dir_t'], 'f2'),
                        ('directional rotational speed |log ratio|', ['global', 'median_abslog', 'dir_r'], 'f2'),
                        ('anisotropic-equation residual |N − 1|', ['global', 'median_absres', 'aniso'], 'f3'),
                        ('τ(wrap) / τ(control)', ['rot_pairs', 'paired_wrap_over_control_median'], 'f2'),
                        ('cut locus median |Δτ| / τ', ['cut_locus', 'median_rel_tau_diff'], 'f3')):
        cell_field_rows.append({'label': '{} · {}'.format(c, lab), 'vals': {k: cf(k, p) for k in KINDS}, 'fmt': fmt})

# ------------------------------------------------------------------ RNL (two-route network)
def rates(path_):
    try:
        t = open(path_).read()
    except OSError:
        return None, None
    fw = re.search(r'regular/forward\s*:\s*([\d.]+)', t)
    ei = re.search(r'either/or\s*:\s*([\d.]+)', t)
    return (float(fw.group(1)) if fw else None), (float(ei.group(1)) if ei else None)


rnl_rows = []
base = {'shape': rates('results/output_3d/Tshape3d_env1_shape_metric_final/success_rate.txt'),
        'lie': rates('results/output_3d/Tshape3d_env1_lie_metric_final/success_rate.txt')}
diag = {}
if args.rnl_diag and os.path.exists(args.rnl_diag):
    for line in open(args.rnl_diag):
        try:
            d = json.loads(line)
            diag[d['tag']] = d
        except Exception:
            pass
for pkg, nm in (('lie_metric', 'lie (anisotropic)'), ('shape_metric', 'shape (blocked)')):
    for mode, mn in (('orig', 'original (chart line)'), ('lie', 'Lie-group curve')):
        name = 'Tshape3d_env1_{}_rnl_{}_sntf'.format(pkg, mode)
        f4, e4 = rates('results/output_3d/{}_ep04000/success_rate.txt'.format(name))
        f5, e5 = rates('results/output_3d/{}_ep05000/success_rate.txt'.format(name))
        d4 = diag.get(name + '_ep04000', {}).get('lie', {})
        b = base[pkg.split('_')[0]]
        rnl_rows.append([nm, mn, ('{:.1f}%'.format(100 * f4) if f4 is not None else 'running'),
                         '{:.1f}%'.format(100 * e4) if e4 is not None else '–',
                         '{:.1f}%'.format(100 * f5) if f5 is not None else 'running',
                         '{:.1f}%'.format(100 * e5) if e5 is not None else '–',
                         '{:.1f}%'.format(100 * b[0]) if b[0] is not None else '–'])
rnl = {'note': ('Shape-NTFields RNL as committed in shape-ntfields/models/metric: probe step ε = 0.02, target slopes '
                'ρ<sub>trans</sub> = −20 and ρ<sub>rot</sub> = −6, slope clip G<sub>max</sub> = 10, weight '
                '10<sup>−4</sup>·clip((epoch − 1500)/2000, 0, 1) per block, contribution capped at 2/3 of the '
                'Eikonal term; applied at the band point x0 with the goal x1 fixed. Two-route network, 5000 epochs, '
                'Tshape3d_env1; evaluated with evaluate_training_3d_batched.py on the standard 1000 cases. "Original" '
                'probes the rotation as a straight line in rotation-vector coordinates; "Lie" turns the body about '
                'the world axis n<sub>rot</sub> through its centre (R → Exp(2πε n<sub>rot</sub>) R). One seed per cell.'),
       'head': ['Eikonal', 'RNL rotation probe', 'forward, ep 4000', 'either, ep 4000', 'forward, ep 5000',
                'either, ep 5000', 'no RNL (forward)'],
       'rows': rnl_rows}

# extended RNL suite (analysis/plan_eval.py, final-epoch RNL models vs no-RNL two-route baselines)
RS = S.get('rnl_suite', {})
rnl_suite_rows = []
for sid, e in RS.items():
    r = e['rates']
    def cell(t):
        v = r.get(t)
        if not v:
            return '–'
        sd = v.get('seeds', {})
        if len(sd) > 1:
            return '{:.1f}% <span style="color:var(--muted)">({})</span>'.format(100 * v['success'], ' / '.join('{:.1f}'.format(100 * sd[k]) for k in sorted(sd)))
        return '{:.1f}%'.format(100 * v['success'])
    def pv(key):
        p = e['pairs'].get(key)
        if not p:
            return '–'
        return '{} vs {} · p {}{}'.format(p['only_A'], p['only_B'], ('{:.0e}'.format(p['p']) if p['p'] < 0.001 else '{:.2f}'.format(p['p'])),
                                          ' ({} seeds)'.format(len(p['seeds'])) if len(p.get('seeds', [])) > 1 else '')
    n = next(iter(r.values()))['n']
    rnl_suite_rows.append([e['label'], n, cell('rnl_shape_none'), cell('rnl_shape_orig'), cell('rnl_shape_lie'), pv('shape: lie vs orig'),
                           cell('rnl_lie_none'), cell('rnl_lie_orig'), cell('rnl_lie_lie'), pv('lie: lie vs orig')])

rnl['suiteHead'] = ['Test set', 'n', 'shape · none', 'shape · orig.', 'shape · Lie', 'shape: Lie vs orig.',
                     'lie · none', 'lie · orig.', 'lie · Lie', 'lie: Lie vs orig.']
rnl['suiteRows'] = rnl_suite_rows

# ------------------------------------------------------------------ text
fpath = os.path.join(R, 'findings.json')
findings = json.load(open(fpath)) if os.path.exists(fpath) else ['Findings are written after all runs finish.']
seeds = {c: {k: len(S['tags'].get(c, {}).get(k, [])) for k in KINDS} for c in S['cells']}
nT = seeds.get('Tshape3d_env1', {})
facts = [('environments', 'Tshape3d_env1 (+ Lshape3d_env3, Ashape3d_env4)'),
         ('seeds on Tshape3d_env1', ' / '.join('{} {}'.format(k, nT.get(k, 0)) for k in KINDS)),
         ('training', '4000 epochs, 800k pairs, same trainer'),
         ('test cases', '{:,} Tshape3d_env1 cases per model'.format(sum(s['n'] for s in sets if s['cell'] == 'Tshape3d_env1'))),
         ('held-out field probes', '{:,} configurations'.format(path(runs('lie')[0], ['global', 'n_probe']) if runs('lie') else 0))]
DATA = {'generated': time.strftime('%Y-%m-%d %H:%M %Z'),
        'lede': ('Three Eikonal formulations trained on the same metric NTFields network and data, compared on planning '
                 '(nine Tshape3d_env1 test sets, including ntrlshape\'s non-linear and long-distance sets, and two '
                 'further environments) and on the travel-time fields they learn: speeds by clearance, which local '
                 'metric each field encodes, and how each treats rotation.'),
        'facts': facts, 'findings': findings, 'sets': sets, 'breakdown': breakdown, 'pairs': pairs,
        'field': field, 'calib': calib, 'metricRows': metric_rows, 'rotRows': rot_rows, 'dual': dual, 'rnl': rnl,
        'cellField': cell_field_rows,
        'cellsNote': 'Same protocol, {} seed(s) per model for Lshape3d_env3 and {} for Ashape3d_env4 (cells show the seed mean); tight-population test sets from ntrlshape.'.format(
            min(seeds.get('Lshape3d_env3', {}).values() or [0]), min(seeds.get('Ashape3d_env4', {}).values() or [0]))}
tpl = open('analysis/report_template.html').read()
html = tpl.replace('__DATA__', json.dumps(DATA, default=float).replace('</', '<\\/'))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, 'w').write(html)
print('wrote', args.out, '({:.0f} KB)'.format(len(html) / 1024))
