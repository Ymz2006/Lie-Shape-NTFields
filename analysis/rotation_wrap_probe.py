"""Does tau respect SO(3) at the rotation-vector seam, or only the chart?

Rotations enter the networks as canonical rotation vectors r = theta n / (2 pi),
theta in [0, pi].  The chart has a seam at theta = pi: (pi - a) n and -(pi - b) n
are a + b apart on SO(3) but about 2 pi apart as vectors, and +pi n and -pi n are
the same pose.  metric_baseline and lie_metric both Fourier-map the raw vector
(sin/cos(rotvec @ B) with real-valued B), so nothing in the architecture glues the
two sides of the seam together, and their losses constrain only |grad tau|, never
its value across the seam.  This script measures what training did there.

The two endpoints of every pair share one position, so tau is pure rotational
travel, and every pose used is collision-free (training clearance labels).
Anchors are (test-set position, random axis n) whose rotations about n by
180 +/- (max delta + flank) deg are all free.

  slide   A fixed SO(3) step delta about n, slid along n across the seam:
          x = Rot((c - delta/2) n), y = Rot((c + delta/2) n).  Every pair is the
          same rotation by delta and the poses move continuously with c, so the
          true tau(c) is continuous; the pair straddles the seam exactly when
          |c - 180| < delta/2.  Reported as tau / (mean tau of the non-straddling
          pairs on the same anchor): a seam-consistent field stays near 1, and
          'edge' is the tau step at the window edges over the typical 1-deg step.
  same    The half-turn Rot(pi n) written as +pi n and as -pi n, against the same
          targets y = Rot((180 -/+ k) n): from one encoding y is k deg away in the
          chart, from the other 360 - k.  k = 0 is tau between two encodings of one
          pose (true value 0), also placed as a percentile of the model's tau over
          the test set's start-goal pairs.
  random  Random body-frame steps y = x Exp(delta u) from x within delta of the
          seam, split by whether the shortest rotation x -> y crosses the seam,
          plus an interior control (x at 30-120 deg).  All pairs are delta apart on
          SO(3), so any gap between groups comes from the chart.  rho is the rank
          correlation of tau with the chart gap among the non-crossing seam pairs
          (the chart stretches tangentially by up to pi/2 near the seam).
  plans   With PLANTAGs, the model's plan_eval records for --dataPath (forward MPPI,
          chart or SE(3) convergence) split by whether the start -> goal shortest
          rotation crosses the seam.  MPPI steps in the chart; with chart
          convergence a seam-consistent field that reaches the goal through its
          other rotation vector counts as not converged, so compare those fields
          on SE(3) convergence too.  'chart line' is how far the straight chart
          segment start -> goal rotates, over the geodesic.

Models labelled <variant>_s<N> are pooled as repeats of <variant> in the figures
and in the closing [variants] table (mean +- sd over repeats).

usage (repo root, inside the container; the clearance checks need a GPU):
  python analysis/rotation_wrap_probe.py --out results/rotation_wrap
  python analysis/rotation_wrap_probe.py --model lie_s1=lie:<ckpt or glob>[:<plantag>[,<plantag>_g]] ...
  python analysis/rotation_wrap_probe.py --model-list <file of --model specs> --out ...
  # positive control: matrix rotation input, so tau is a function on SO(3) by construction
  python analysis/rotation_wrap_probe.py --out results/rotation_wrap_matrix_control --model \\
    'mat_s1=twist:Experiments/3dshape_lie_twist/Tshape3d_env1_lie_twist_iso_comb_id_mat_s1/Model_Epoch_04000_*.pt:iso_comb_id_mat_s1'
"""
import os
import re
import sys
import glob
import math
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append('.')
from analysis.common import Scene, load_model, so3_exp, so3_log, save_json, TWO_PI  # noqa: E402

DEFAULT_MODELS = (
    ['metric_s{0}=metric:Experiments/3dshape_metric_baseline/Tshape3d_env1_metric_baseline_s{0}/'
     'Model_Epoch_04000_*.pt:metric_s{0},metric_s{0}_g'.format(s) for s in (1, 2, 3)]
    + ['lie_s{0}=lie:Experiments/3dshape_lie_metric/Tshape3d_env1_lie_metric_rsqrtN_s{0}/'
       'Model_Epoch_04000_*.pt:lie_rsqrtN_s{0},lie_rsqrtN_s{0}_g'.format(s) for s in (1, 2, 3)])
# Categorical slots of the dataviz reference palette, assigned in order.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
GROUPS = ('interior', 'seam, no crossing', 'seam, crossing')

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model', action='append', metavar='LABEL=KIND:CKPT[:PLANTAG[,PLANTAG]]',
                help='repeatable; KIND as in analysis/common.py MODEL_PKGS, CKPT may be a glob, '
                     'PLANTAG is a results/compare/plan/<tag> folder (chart or SE(3) convergence, read '
                     'from its summary).  Labels <group>_s<N> are pooled as repeats of <group> '
                     '(default: metric_baseline and lie_metric rsqrtN on Tshape3d_env1, seeds 1-3, epoch 4000)')
ap.add_argument('--model-list', help='file with one --model spec per line (# comments)')
ap.add_argument('--dataPath', default='testing_data/3dshape/Tshape3d_env1',
                help='test set: scene, anchor positions and the plan split')
ap.add_argument('--planRoot', default='results/compare/plan')
ap.add_argument('--out', default='results/rotation_wrap')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--deltas', default='4,10,20', help='SO(3) step sizes (even degrees)')
ap.add_argument('--flank', type=int, default=20, help='slide range past the straddling window, each side (deg)')
ap.add_argument('--anchors', type=int, default=128)
ap.add_argument('--pairs', type=int, default=500, help='random pairs per group per delta')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

DELTAS = [int(d) for d in args.deltas.split(',')]
assert all(d > 0 and d % 2 == 0 for d in DELTAS), '--deltas must be even degrees'
HALF_ARC = max(DELTAS) + args.flank
os.makedirs(args.out, exist_ok=True)
gen = torch.Generator().manual_seed(args.seed)


def parse_model(spec):
    label, rest = spec.split('=', 1)
    parts = rest.split(':')
    hits = sorted(glob.glob(parts[1]))
    if not hits:
        raise SystemExit('no checkpoint matches ' + parts[1])
    return {'label': label, 'group': re.sub(r'_s\d+$', '', label), 'kind': parts[0], 'ckpt': hits[-1],
            'tags': parts[2].split(',') if len(parts) > 2 and parts[2] else []}


specs = list(args.model or [])
if args.model_list:
    specs += [ln.strip() for ln in open(args.model_list) if ln.strip() and not ln.lstrip().startswith('#')]
MODELS = [parse_model(s) for s in (specs or DEFAULT_MODELS)]
GROUP_NAMES = list(dict.fromkeys(m['group'] for m in MODELS))


# --------------------------------------------------------------------------- SO(3)
def unit(n):
    v = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    return v / v.norm(dim=1, keepdim=True)


def axis_chart(n, deg):
    """Stored rotation vectors (turns) of Rot(deg about n): n (A,3), deg (K,) -> (A,K,3).
    Angles past 180 come out as -(360 - deg) about n, the way the data stores them."""
    th = torch.as_tensor(deg, dtype=torch.float64) * (math.pi / 180)
    th = torch.where(th > math.pi, th - TWO_PI, th)
    return n[:, None, :] * th[None, :, None] / TWO_PI


def crosses_seam(r0, r1, k=65):
    """(B,) bool: the shortest rotation from r0 to r1 (turns, (B,3)) passes the seam,
    i.e. its canonical rotation vector jumps (by ~2 pi) somewhere along the way."""
    R0 = so3_exp(TWO_PI * torch.as_tensor(r0, dtype=torch.float64))
    w = so3_log(R0.transpose(-1, -2) @ so3_exp(TWO_PI * torch.as_tensor(r1, dtype=torch.float64)))
    t = torch.linspace(0, 1, k, dtype=torch.float64)
    r = so3_log(R0[:, None] @ so3_exp(t[None, :, None] * w[:, None, :]))
    return (r[:, 1:] - r[:, :-1]).norm(dim=-1).max(1).values > math.pi / 2


# --------------------------------------------------------------------------- pairs
def free(scene, p, r):
    """Collision-free mask for positions p (..., 3) broadcast against rotations r (..., 3)."""
    shape = r.shape[:-1]
    q = torch.cat([p.expand(*shape, 3), r], -1).reshape(-1, 6)
    return scene.clearance(q)['free'].reshape(shape)


def find_anchors(scene, pos, batch=256, max_batches=40):
    """(position, axis) pairs whose rotations about the axis by 180 +/- HALF_ARC deg
    (every 1 deg) are all collision-free."""
    arc = torch.arange(180 - HALF_ARC, 180 + HALF_ARC + 1, dtype=torch.float64)
    gate = arc[[HALF_ARC, 0, -1]]
    P, N, tried = [], [], 0
    for _ in range(max_batches):
        p = pos[torch.randint(len(pos), (batch,), generator=gen)][:, None, :]
        n = unit(batch)
        tried += batch
        ok = free(scene, p, axis_chart(n, gate)).all(1)
        p, n = p[ok], n[ok]
        if len(n):
            ok = free(scene, p, axis_chart(n, arc)).all(1)
            P.append(p[ok, 0])
            N.append(n[ok])
        if sum(len(x) for x in N) >= args.anchors:
            break
    return torch.cat(P)[:args.anchors], torch.cat(N)[:args.anchors], tried


def random_pairs(scene, pos, delta, batch=1024, max_batches=60):
    """Collision-free pairs (x, y = x Exp(delta u)) per group; the shortest rotation
    between them (7 samples) is free too."""
    d = math.radians(delta)
    t = torch.linspace(0, 1, 7, dtype=torch.float64)
    got = {g: [] for g in GROUPS}
    for _ in range(max_batches):
        p = pos[torch.randint(len(pos), (batch,), generator=gen)]
        near = torch.rand(batch, generator=gen) < 0.5
        lo = torch.where(near, math.pi - d, math.radians(30)).double()
        hi = torch.where(near, math.pi, math.radians(120)).double()
        th = lo + (hi - lo) * torch.rand(batch, generator=gen, dtype=torch.float64)
        Rx = so3_exp(th[:, None] * unit(batch))
        u = unit(batch)
        path = so3_log(Rx[:, None] @ so3_exp(t[None, :, None] * (d * u)[:, None, :])) / TWO_PI
        ok = free(scene, p[:, None, :], path).all(1)
        x, y = path[:, 0], path[:, -1]
        cross = crosses_seam(x, y)
        for g, m in zip(GROUPS, (~near, near & ~cross, near & cross)):
            m = m & ok
            got[g].append(torch.cat([p[m], x[m], p[m], y[m]], 1))
        if all(sum(len(a) for a in v) >= args.pairs for v in got.values()):
            break
    return {g: torch.cat(v)[:args.pairs] for g, v in got.items()}


@torch.no_grad()
def tau_of(fn, X, chunk=16384):
    shape = X.shape[:-1]
    X = X.reshape(-1, 12).float()
    out = [fn.TravelTimes(X[s:s + chunk].to(args.device)).double().cpu() for s in range(0, len(X), chunk)]
    return torch.cat(out).reshape(shape)


def q(a, p):
    return float(np.quantile(np.asarray(a, dtype=np.float64), p))


def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


# --------------------------------------------------------------------------- setup
meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))


def _resolve(path):
    return path if os.path.exists(path) else os.path.join('datasets/3dshape', os.path.basename(path))


scene = Scene(_resolve(meta['env_obj']), _resolve(meta['shape_obj']), device=args.device, seed=args.seed)
test = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
pos = torch.tensor(np.concatenate([test[:, 0:3], test[:, 6:9]]))

P, N, tried = find_anchors(scene, pos)
print('anchors: {} of {} candidates have a free {}-deg arc around the half-turn'.format(
    len(P), tried, 2 * HALF_ARC), flush=True)
if len(P) == 0:
    raise SystemExit('no anchors found; lower --flank or --deltas')

slide_X, slide_c = {}, {}
for delta in DELTAS:
    h = delta // 2
    c = torch.arange(-(h + args.flank), h + args.flank, dtype=torch.float64) + 180.5
    pp = P[:, None, :].expand(-1, len(c), -1)
    slide_X[delta] = torch.cat([pp, axis_chart(N, c - h), pp, axis_chart(N, c + h)], -1)
    slide_c[delta] = c

K = torch.arange(0, HALF_ARC + 1, dtype=torch.float64)
plus = axis_chart(N, [180.0])                      # +pi n (turns: 0.5 n)
y_lo = axis_chart(N, 180 - K)                      # Rot((180 - k) n), stored on the +n side
pK = P[:, None, :].expand(-1, len(K), -1)
same_near = [torch.cat([pK, plus.expand_as(y_lo), pK, y_lo], -1),       # chart gap k
             torch.cat([pK, -plus.expand_as(y_lo), pK, -y_lo], -1)]     # mirrored: -pi n vs Rot((180 + k) n)
same_far = [torch.cat([pK, -plus.expand_as(y_lo), pK, y_lo], -1),       # chart gap 360 - k
            torch.cat([pK, plus.expand_as(y_lo), pK, -y_lo], -1)]

rand_X = {}
for delta in DELTAS:
    rand_X[delta] = random_pairs(scene, pos, delta)
    print('random pairs, delta {}: {}'.format(delta, {g: len(v) for g, v in rand_X[delta].items()}), flush=True)

np.savez(os.path.join(args.out, 'pairs.npz'), anchor_pos=P.numpy(), anchor_axis=N.numpy(),
         **{'random_d{}_{}'.format(d, g.replace(', ', '_').replace(' ', '_')): v.numpy()
            for d in DELTAS for g, v in rand_X[d].items()})

# --------------------------------------------------------------------------- evaluate
summary = {'args': vars(args), 'n_anchors': len(P), 'anchor_candidates': tried,
           'models': {m['label']: {'kind': m['kind'], 'checkpoint': m['ckpt']} for m in MODELS}}
curves = {}
for m in MODELS:
    _, fn = load_model(m['kind'], m['ckpt'], args.device)
    S = summary['models'][m['label']]
    cv = curves[m['label']] = {'slide': {}, 'random': {}}

    S['slide'] = {}
    for delta in DELTAS:
        tau = tau_of(fn, slide_X[delta])                                  # (A, C)
        c = slide_c[delta]
        win = (c - 180).abs() < delta / 2
        flank = tau[:, ~win].mean(1)
        ratio = tau[:, win].mean(1) / flank
        step = tau.diff(dim=1).abs()
        edge = win[1:] != win[:-1]
        edge_ratio = step[:, edge].mean(1) / step[:, ~edge].median(1).values.clamp_min(1e-9)
        S['slide'][delta] = {
            'tau_flank_median': q(flank, 0.5), 'tau_window_median': q(tau[:, win].mean(1), 0.5),
            'ratio_median': q(ratio, 0.5), 'ratio_q25': q(ratio, 0.25), 'ratio_q75': q(ratio, 0.75),
            'frac_ratio_gt_1p25': float((ratio > 1.25).double().mean()),
            'edge_step_over_typical_median': q(edge_ratio, 0.5)}
        cv['slide'][delta] = (tau / flank[:, None]).numpy()                # (A, C), pooled per group

    near = 0.5 * (tau_of(fn, same_near[0]) + tau_of(fn, same_near[1]))  # (A, K)
    far = 0.5 * (tau_of(fn, same_far[0]) + tau_of(fn, same_far[1]))
    tau_test = tau_of(fn, torch.tensor(test))
    S['same'] = {int(k): {'tau_near_median': q(near[:, i], 0.5), 'tau_far_median': q(far[:, i], 0.5)}
                 for i, k in enumerate(K.tolist()) if k in (0, 2, 5, 10, 20, 30, HALF_ARC)}
    S['same']['test_tau_median'] = q(tau_test, 0.5)
    S['same']['test_tau_max'] = float(tau_test.max())
    S['same']['same_pose_tau_percentile_of_test'] = float((tau_test < q(far[:, 0], 0.5)).double().mean() * 100)
    cv['same'] = {'near': near.numpy(), 'far': far.numpy()}

    S['random'] = {}
    for delta in DELTAS:
        R, taus = {}, {}
        for g, X in rand_X[delta].items():
            if not len(X):
                continue
            taus[g] = tau_of(fn, X)
            gap = (X[:, 9:12] - X[:, 3:6]).norm(dim=1) * 360
            R[g] = {'n': len(X), 'tau_median': q(taus[g], 0.5), 'tau_q25': q(taus[g], 0.25),
                    'tau_q75': q(taus[g], 0.75), 'chart_gap_deg_median': q(gap, 0.5)}
            cv['random'].setdefault(delta, {})[g] = (gap.numpy(), taus[g].numpy())
        if all(g in taus for g in GROUPS):
            R['crossing_over_no_crossing'] = R[GROUPS[2]]['tau_median'] / R[GROUPS[1]]['tau_median']
            R['no_crossing_over_interior'] = R[GROUPS[1]]['tau_median'] / R[GROUPS[0]]['tau_median']
            R['spearman_tau_vs_chart_gap_no_crossing'] = spearman(
                taus[GROUPS[1]], torch.as_tensor(cv['random'][delta][GROUPS[1]][0]))
        S['random'][delta] = R
    del fn
    torch.cuda.empty_cache()
    print('evaluated', m['label'], flush=True)

# --------------------------------------------------------------------------- plans
set_name = os.path.basename(os.path.normpath(args.dataPath))
r0, r1 = torch.tensor(test[:, 3:6]), torch.tensor(test[:, 9:12])
plan_cross = crosses_seam(r0, r1).numpy()
plan_geo = so3_log(so3_exp(TWO_PI * r0).transpose(-1, -2) @ so3_exp(TWO_PI * r1)).norm(dim=-1).numpy()
Rl = so3_exp(TWO_PI * (r0[:, None] + torch.linspace(0, 1, 65, dtype=torch.float64)[None, :, None] * (r1 - r0)[:, None]))
chart_line = so3_log(Rl[:, :-1].transpose(-1, -2) @ Rl[:, 1:]).norm(dim=-1).sum(1).numpy() / np.maximum(plan_geo, 1e-3)
plan_geo = plan_geo * 180 / math.pi
summary['plans'] = {'set': set_name, 'n_cases': len(test), 'n_crossing': int(plan_cross.sum()), 'models': {}}
for name, g in (('crossing', plan_cross), ('no_crossing', ~plan_cross)):
    summary['plans'][name] = {'geo_deg_median': q(plan_geo[g], 0.5) if g.any() else None,
                              'chart_line_over_geodesic_median': q(chart_line[g], 0.5) if g.any() else None}
GEO_BINS = (0, 90, 120, 150, 181)


def plan_stats(cr, geo, succ, conv, coll, detour):
    out = {'all': {'n': len(succ), 'success': float(succ.mean())}}
    for name, g in (('crossing', cr), ('no_crossing', ~cr)):
        out[name] = {'n': int(g.sum()), 'success': float(succ[g].mean()) if g.any() else None,
                     'collision': float(coll[g].mean()) if g.any() else None,
                     'nonconv': float((~conv[g]).mean()) if g.any() else None,
                     'rot_path_over_geodesic_median_success': q(detour[g & succ], 0.5) if (g & succ).any() else None}
    out['by_geodesic_deg'] = {}
    for lo, hi in zip(GEO_BINS[:-1], GEO_BINS[1:]):
        b = (geo >= lo) & (geo < hi)
        out['by_geodesic_deg']['{}-{}'.format(lo, min(hi, 180))] = {
            name: {'n': int((b & g).sum()), 'success': float(succ[b & g].mean()) if (b & g).any() else None}
            for name, g in (('crossing', cr), ('no_crossing', ~cr))}
    return out


CONVS = ('chart', 'se3')                  # plan_eval convergence: in the chart / on SE(3) (--group-conv)
pooled = {}
for m in MODELS:
    for tag in m['tags']:
        path = os.path.join(args.planRoot, tag, set_name + '.json')
        if not os.path.exists(path):
            print('no plan records at', path)
            continue
        js = json.load(open(path))
        conv_kind = 'se3' if js['summary'].get('group_conv') else 'chart'
        if js['summary'].get('planner') not in (None, 'mppi'):
            print('skipping {}: planner {}'.format(tag, js['summary']['planner']))
            continue
        rec = js['records']
        cases = np.array([r['case'] for r in rec])
        cols = (plan_cross[cases], plan_geo[cases], np.array([r['success'] for r in rec]),
                np.array([r['converged'] for r in rec]), np.array([r['collision'] for r in rec]),
                np.array([r['rot_len_geo'] / max(r['straight_rot_geo'], 1e-3) for r in rec]))
        out = plan_stats(*cols)
        out['tag'] = tag
        if os.path.normpath(js['summary'].get('checkpoint', '')) != os.path.normpath(m['ckpt']):
            out['note'] = 'plan records were made with {}'.format(js['summary'].get('checkpoint'))
        summary['plans']['models'].setdefault(m['label'], {})[conv_kind] = out
        pooled.setdefault((m['group'], conv_kind), []).append(cols)
summary['plans']['groups'] = {}
for (grp, conv_kind), runs in pooled.items():
    summary['plans']['groups'].setdefault(grp, {})[conv_kind] = plan_stats(*[np.concatenate(c) for c in zip(*runs)])

# Per-group field summary: mean and spread over the group's repeats.
D_MID = DELTAS[len(DELTAS) // 2]


def agg(vals):
    v = np.array([x for x in vals if x is not None], dtype=np.float64)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0, 'n': int(len(v))} if len(v) else None


summary['groups'] = {}
for grp in GROUP_NAMES:
    labels = [m['label'] for m in MODELS if m['group'] == grp]
    S = [summary['models'][lb] for lb in labels]
    P = [summary['plans']['models'].get(lb, {}) for lb in labels]
    summary['groups'][grp] = {
        'labels': labels,
        'slide_ratio': agg([s['slide'][D_MID]['ratio_median'] for s in S]),
        'slide_tau_off': agg([s['slide'][D_MID]['tau_flank_median'] for s in S]),
        'same_pose_tau': agg([s['same'][0]['tau_far_median'] for s in S]),
        'same_pose_pct_of_test': agg([s['same']['same_pose_tau_percentile_of_test'] for s in S]),
        'test_tau_median': agg([s['same']['test_tau_median'] for s in S]),
        'cross_over_no_cross': agg([s['random'][D_MID].get('crossing_over_no_crossing') for s in S]),
        'no_cross_over_interior': agg([s['random'][D_MID].get('no_crossing_over_interior') for s in S]),
        **{'success_' + c: agg([p[c]['all']['success'] for p in P if c in p]) for c in CONVS},
        **{'success_crossing_' + c: agg([p[c]['crossing']['success'] for p in P if c in p]) for c in CONVS},
        **{'success_no_crossing_' + c: agg([p[c]['no_crossing']['success'] for p in P if c in p]) for c in CONVS}}


# --------------------------------------------------------------------------- report
def pct(v):
    return '  -  ' if v is None else '{:5.1f}'.format(100 * v)


lines = ['anchors: {} (free {}-deg arc around the half-turn, from {} candidates)'.format(
    len(P), 2 * HALF_ARC, tried), '']
lines.append('[slide] tau of a fixed rotation step slid across the seam (medians over anchors)')
lines.append('        off / on = mean tau while the pair is off / straddling the seam; ratio = on / off per anchor')
lines.append('        (1 = seam-consistent); edge = tau jump at the window edges / typical 1-deg change')
for m in MODELS:
    s = summary['models'][m['label']]['slide']
    lines.append('  {:<12}'.format(m['label']) + '   '.join(
        'd={:>2}: off {:.3f} on {:.3f} ratio {:5.1f} edge {:6.0f}'.format(
            d, s[d]['tau_flank_median'], s[d]['tau_window_median'], s[d]['ratio_median'],
            s[d]['edge_step_over_typical_median'])
        for d in DELTAS))
lines += ['', '[same] half-turn written as +pi n vs -pi n, target Rot((180 -/+ k) n); medians over anchors',
          '       near/far = tau from the encoding on the target\'s side (chart gap k) / the other one (360 - k);',
          '       true value is the same for both, and 0 at k = 0']
for m in MODELS:
    s = summary['models'][m['label']]['same']
    lines.append('  {:<12}'.format(m['label']) + '  '.join(
        'k={:>2}: {:.3f}/{:.3f}'.format(k, s[k]['tau_near_median'], s[k]['tau_far_median']) for k in (0, 5, 20))
        + '  | same pose > {:.0f}% of test start-goal taus (median {:.3f}, max {:.3f})'.format(
            s['same_pose_tau_percentile_of_test'], s['test_tau_median'], s['test_tau_max']))
lines += ['', '[random] steps of size delta, median tau per group (all pairs are delta apart on SO(3)), chart gap in deg',
          '         rho = rank corr(tau, chart gap) among non-crossing seam pairs (0 = follows SO(3), not the chart)']
for m in MODELS:
    s = summary['models'][m['label']]['random']
    for d in DELTAS:
        r = s[d]
        lines.append('  {:<12} d={:>2}: '.format(m['label'], d) + '  '.join(
            '{} {:.3f} (gap {:5.1f})'.format(g, r[g]['tau_median'], r[g]['chart_gap_deg_median'])
            for g in GROUPS if g in r) + '  cross/no-cross {:5.1f}  no-cross/interior {:.2f}  rho {:.2f}'.format(
            r.get('crossing_over_no_crossing', float('nan')), r.get('no_crossing_over_interior', float('nan')),
            r.get('spearman_tau_vs_chart_gap_no_crossing', float('nan'))))
pl = summary['plans']
lines += ['', '[plans] {}: {} of {} start-goal rotations cross the seam'.format(pl['set'], pl['n_crossing'], pl['n_cases'])]
for name in ('crossing', 'no_crossing'):
    if pl[name]['geo_deg_median'] is not None:
        lines.append('        {:<11}: median geodesic {:.0f} deg, chart line rotates {:.2f}x the geodesic'.format(
            name, pl[name]['geo_deg_median'], pl[name]['chart_line_over_geodesic_median']))
def plan_lines(name, r):
    c, nc = r['crossing'], r['no_crossing']
    det = '{:.2f} / {:.2f}'.format(c['rot_path_over_geodesic_median_success'] or float('nan'),
                                   nc['rot_path_over_geodesic_median_success'] or float('nan'))
    out = ['  {:<24} success {} | cross {} vs other {} | collision {} vs {} | nonconv {} vs {} | '
           'rot path / geodesic {}'.format(name, pct(r['all']['success']), pct(c['success']), pct(nc['success']),
                                           pct(c['collision']), pct(nc['collision']), pct(c['nonconv']),
                                           pct(nc['nonconv']), det)]
    out.append('  {:<24} by geodesic: '.format('') + '  '.join(
        '{}: {} (n {}) vs {} (n {})'.format(b, pct(v['crossing']['success']), v['crossing']['n'],
                                            pct(v['no_crossing']['success']), v['no_crossing']['n'])
        for b, v in r['by_geodesic_deg'].items()))
    if 'note' in r:
        out.append('  {:<24} NOTE {}'.format('', r['note']))
    return out


for label, convs in pl['models'].items():
    for conv_kind, r in convs.items():
        lines += plan_lines('{} [{}]'.format(label, conv_kind), r)
lines += ['', '  pooled over repeats:']
for grp, convs in pl['groups'].items():
    for conv_kind, r in convs.items():
        lines += plan_lines('{} [{}]'.format(grp, conv_kind), r)


def ms(a, scale=1.0, fmt='{:.1f}'):
    if a is None:
        return '   -   '
    return (fmt + ' +- ' + fmt).format(scale * a['mean'], scale * a['sd']) if a['n'] > 1 else fmt.format(scale * a['mean'])


lines += ['', '[variants] mean +- sd over repeats; field numbers at d = {} deg; plan success on {} (forward MPPI)'.format(
    D_MID, pl['set']),
    '  seam ratio  = slide test, tau while straddling / off the seam (1 = seam-consistent)',
    '  same pose   = tau between the two vectors of one half-turn (true 0), and the share of test start-goal taus below it',
    '  cross/no    = random test, crossing / non-crossing seam pairs; no/inter = non-crossing seam pairs / interior pairs',
    '  {:<12} {:>2}  {:>14}  {:>16}  {:>6}  {:>11}  {:>9} | {:>13} {:>13} | {:>13} {:>13}'.format(
        'variant', 'n', 'seam ratio', 'same pose', '%test', 'cross/no', 'no/inter',
        'success chart', 'success SE3', 'cross chart', 'cross SE3')]
for grp, g in summary['groups'].items():
    lines.append('  {:<12} {:>2}  {:>14}  {:>16}  {:>6}  {:>11}  {:>9} | {:>13} {:>13} | {:>13} {:>13}'.format(
        grp, len(g['labels']), ms(g['slide_ratio']), ms(g['same_pose_tau'], fmt='{:.3f}'),
        ms(g['same_pose_pct_of_test'], fmt='{:.0f}').split(' ')[0], ms(g['cross_over_no_cross']),
        ms(g['no_cross_over_interior'], fmt='{:.2f}').split(' ')[0],
        ms(g['success_chart'], 100), ms(g['success_se3'], 100),
        ms(g['success_crossing_chart'], 100), ms(g['success_crossing_se3'], 100)))
save_json(summary, os.path.join(args.out, 'summary.json'))
report = '\n'.join(lines)
print('\n' + report)
open(os.path.join(args.out, 'report.txt'), 'w').write(report + '\n')


# --------------------------------------------------------------------------- figures
def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


# One colour per variant (fixed slot order); past eight, the slots repeat dashed.
group_style = {grp: dict(color=COLORS[i % len(COLORS)], ls=('-', '--')[(i // len(COLORS)) % 2], lw=1.5)
               for i, grp in enumerate(GROUP_NAMES)}


def pooled_curve(grp, *keys):
    arrs = []
    for m in MODELS:
        if m['group'] == grp:
            v = curves[m['label']]
            for k in keys:
                v = v[k]
            arrs.append(v)
    return np.concatenate(arrs)


fig, axes = plt.subplots(1, len(DELTAS), figsize=(4.6 * len(DELTAS), 3.8), squeeze=False)
for ax, d in zip(axes[0], DELTAS):
    c = slide_c[d].numpy() - 180
    ax.axvspan(-d / 2, d / 2, color=GRID, lw=0)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    for grp in GROUP_NAMES:
        ax.plot(c, np.median(pooled_curve(grp, 'slide', d), 0), label=grp, **group_style[grp])
    ax.set_yscale('log')
    ax.set_title('{}-deg step slid across the seam'.format(d), fontsize=10, color=INK)
    ax.set_xlabel('pair centre minus 180 (deg); shaded = pair straddles the seam', fontsize=8, color=MUTED)
    style(ax)
axes[0][0].set_ylabel('tau / tau of the same step off the seam\n(median over anchors and repeats)', fontsize=8, color=MUTED)
axes[0][-1].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'slide.png'), dpi=150)
plt.close(fig)

ncol = min(4, len(GROUP_NAMES))
nrow = int(math.ceil(len(GROUP_NAMES) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
for ax, grp in zip(axes.flat, GROUP_NAMES):
    for j, (s, name) in enumerate((('near', 'encoding on target side (gap k)'),
                                   ('far', 'opposite encoding (gap 360 - k)'))):
        lo, mid, hi = np.quantile(pooled_curve(grp, 'same', s), [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(K.numpy(), lo, hi, color=COLORS[j], alpha=0.18, lw=0)
        ax.plot(K.numpy(), mid, color=COLORS[j], lw=1.5, label=name)
    ax.set_title(grp, fontsize=10, color=INK)
    ax.set_xlabel('target rotation from the half-turn, k (deg)', fontsize=8, color=MUTED)
    style(ax)
for ax in list(axes.flat)[len(GROUP_NAMES):]:
    ax.axis('off')
for row in axes:
    row[0].set_ylabel('tau (median, IQR)', fontsize=8, color=MUTED)
axes[0][0].legend(fontsize=7, frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'same_pose.png'), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(len(GROUP_NAMES), len(DELTAS), figsize=(4.0 * len(DELTAS), 2.4 * len(GROUP_NAMES)),
                         squeeze=False, sharex='col')
for i, grp in enumerate(GROUP_NAMES):
    for j, d in enumerate(DELTAS):
        ax = axes[i][j]
        for k, g in enumerate((GROUPS[1], GROUPS[2], GROUPS[0])):
            pts = [curves[m['label']]['random'][d][g] for m in MODELS
                   if m['group'] == grp and g in curves[m['label']]['random'].get(d, {})]
            if pts:
                ax.scatter(np.concatenate([p[0] for p in pts]), np.concatenate([p[1] for p in pts]),
                           s=4, color=COLORS[k], alpha=0.35, lw=0, label=g)
        ax.set_title('{}, {}-deg steps'.format(grp, d), fontsize=9, color=INK)
        style(ax)
        if i == len(GROUP_NAMES) - 1:
            ax.set_xlabel('chart gap |r_x - r_y| (deg)', fontsize=8, color=MUTED)
        if j == 0:
            ax.set_ylabel('tau', fontsize=8, color=MUTED)
axes[0][-1].legend(fontsize=7, frameon=False, markerscale=2)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'random.png'), dpi=150)
plt.close(fig)
print('wrote', args.out)
