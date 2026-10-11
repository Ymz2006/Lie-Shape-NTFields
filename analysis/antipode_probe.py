"""Why do SO(3)-smooth rotation encodings plan worse than the rotation-vector chart?

Hypothesis.  With a smooth encoding (rot6, euler6, matrix, atlas) tau(., goal) is a
smooth function on SO(3), which is compact: it must have a maximum, and the
maximum of the true travel time is the goal's cut locus, the half-turns away from
it.  A smooth function has zero gradient at its maximum, so the Eikonal residual
N = sqrt(S_d^2 |grad_v tau|^2 + S_a^2 |grad_w tau|^2) cannot be 1 there; the
network rounds the ridge into a flat cap whose width is set by its bandwidth.
In the chart the goal's antipode is not a maximum: tau keeps growing up to the
seam at |rotvec| = pi (a boundary, where a non-zero gradient is allowed), and the
pairs that would need the ridge are exactly the seam-crossing ones.  The forward
MPPI planner weights its 50 samples by exp(-50 (10 tau_1 + tau_5)); in a flat cap
the weights are uniform and the executed step is the mean of zero-mean noise, so
it stalls.

Measurements (all on the stored checkpoints, nothing is retrained):
  arc     At free anchors (position + axis whose full turn is collision-free), the
          start rotates about a body axis by theta in [0, 360) deg away from a
          random goal rotation: tau(theta), the rotational Eikonal term
          S_a |grad_w tau| and the residual N at the start.  Reported per model as
          medians over anchors, and the 'cap' = half-width around 180 deg where
          the median N < 0.5.
  pairs   Random free pairs (same training-style sampling, relative rotation theta
          uniform in [0, 180] deg): N at both endpoints binned by theta, and by
          the start's distance to the chart seam (|rotvec| in deg).
  mppi    The evaluator's forward MPPI (no collision check) on the test cases with
          relative rotation > --big-deg: per step the geodesic angle and position
          distance to the goal.  Reported: steps until the angle first drops below
          --escape-deg, the executed step length over the first 20 iterations (as a
          fraction of STEP), SE(3) convergence within 200 steps, and the share of
          runs whose final angle is still > --big-deg.

usage (repo root, inside the container):
  python analysis/antipode_probe.py --model-list results/antipode_probe/models.txt --out results/antipode_probe
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
from analysis.common import (Scene, load_model, so3_exp, so3_log, target_speeds, body_rot_grad,  # noqa: E402
                             geodesic_angle, tau_and_grad, save_json, TWO_PI)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model', action='append', metavar='LABEL=KIND:CKPT', help='repeatable; labels <group>_s<N> are pooled')
ap.add_argument('--model-list', help='file with one --model spec per line (# comments)')
ap.add_argument('--dataPath', default='testing_data/3dshape/Tshape3d_env1')
ap.add_argument('--out', default='results/antipode_probe')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--anchors', type=int, default=48)
ap.add_argument('--pairs', type=int, default=6000)
ap.add_argument('--big-deg', type=float, default=150.0)
ap.add_argument('--escape-deg', type=float, default=120.0)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--skip', default='', help='comma list of parts to skip: arc,pairs,mppi')
args = ap.parse_args()
SKIP = set(s for s in args.skip.split(',') if s)
os.makedirs(args.out, exist_ok=True)
gen = torch.Generator().manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DIM = 6
STEP, TEMP, SAMPLE_NUM, HORIZON = 0.015, 50.0, 50, 5
ARC_DEG = np.arange(0, 360, 5)


def parse_model(spec):
    label, rest = spec.split('=', 1)
    kind, ck = rest.split(':', 1)
    hits = sorted(glob.glob(ck))
    if not hits:
        raise SystemExit('no checkpoint matches ' + ck)
    return {'label': label, 'group': re.sub(r'_s\d+$', '', label), 'kind': kind, 'ckpt': hits[-1]}


specs = list(args.model or [])
if args.model_list:
    specs += [ln.strip() for ln in open(args.model_list) if ln.strip() and not ln.lstrip().startswith('#')]
if not specs:
    raise SystemExit('give --model or --model-list')
MODELS = [parse_model(s) for s in specs]
GROUPS = list(dict.fromkeys(m['group'] for m in MODELS))
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


# --------------------------------------------------------------------------- SO(3)
def unit(n):
    v = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    return v / v.norm(dim=1, keepdim=True)


def random_rot(n, lo=0.0, hi=math.pi):
    return unit(n) * (lo + (hi - lo) * torch.rand(n, 1, generator=gen, dtype=torch.float64))


def compose(r_base, r_body):
    """rotvec (turns) of exp(r_base) exp(r_body); both (...,3) in turns."""
    R = so3_exp(TWO_PI * r_base) @ so3_exp(TWO_PI * r_body)
    return so3_log(R) / TWO_PI


# --------------------------------------------------------------------------- scene
meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))


def _resolve(path):
    return path if os.path.exists(path) else os.path.join('datasets/3dshape', os.path.basename(path))


scene = Scene(_resolve(meta['env_obj']), _resolve(meta['shape_obj']), device=args.device, seed=args.seed)
test = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)


def speeds_of(q):
    """Training speed targets (S_dist, S_angle) and free mask of configs q (N,6) turns."""
    c = scene.clearance(q)
    S_d, S_a, _ = target_speeds(c['dist'], c['angle'])
    return S_d, S_a, c['free']


def residual_at(fn, X, S_d, S_a, which=0):
    """Lie Eikonal residual of endpoint `which` of pairs X (N,12): N, S_d|g_v|, S_a|g_w|, tau."""
    tau, g = tau_and_grad(fn, X)
    o = which * DIM
    gv = g[:, o:o + 3].norm(dim=1)
    gw = body_rot_grad(X[:, o + 3:o + 6], g[:, o + 3:o + 6]).norm(dim=1)
    tv, tw = S_d.double() * gv, S_a.double() * gw
    return torch.sqrt(tv**2 + tw**2), tv, tw, tau


summary = {'args': vars(args), 'models': {m['label']: {'kind': m['kind'], 'checkpoint': m['ckpt']} for m in MODELS}}
loaded = {}


def get(m):
    if m['label'] not in loaded:
        loaded.clear()
        torch.cuda.empty_cache()
        loaded[m['label']] = load_model(m['kind'], m['ckpt'], args.device)[1]
    return loaded[m['label']]


def q(a, p):
    return float(np.quantile(np.asarray(a, dtype=np.float64), p))


# =========================================================================== arc
arc = {}
if 'arc' not in SKIP:
    P, N_ax, R_goal, S_arc = [], [], [], []
    tried = 0
    th = torch.as_tensor(ARC_DEG, dtype=torch.float64) * math.pi / 180 / TWO_PI      # turns
    for _ in range(60):
        n = 256
        tried += n
        p = scene.sample_configs(n, gen)[:, :3].double()
        ax = unit(n)
        rg = random_rot(n) / TWO_PI                                     # goal rotation (turns)
        r_arc = compose(rg[:, None, :].expand(-1, len(th), -1), ax[:, None, :] * th[None, :, None])
        cfg = torch.cat([p[:, None, :].expand(-1, len(th), -1), r_arc], -1).reshape(-1, 6)
        S_d, S_a, free = speeds_of(cfg)
        ok = free.reshape(n, -1).all(1)
        P.append(p[ok]); N_ax.append(ax[ok]); R_goal.append(rg[ok])
        S_arc.append(torch.stack([S_d.reshape(n, -1)[ok], S_a.reshape(n, -1)[ok]], -1))
        if sum(len(x) for x in P) >= args.anchors:
            break
    P, N_ax, R_goal = torch.cat(P)[:args.anchors], torch.cat(N_ax)[:args.anchors], torch.cat(R_goal)[:args.anchors]
    S_arc = torch.cat(S_arc)[:args.anchors]
    print('arc anchors: {} of {} candidates have a free full turn'.format(len(P), tried), flush=True)
    summary['arc'] = {'n_anchors': len(P), 'candidates': tried, 'deg': ARC_DEG.tolist(),
                      'S_dist_median': q(S_arc[..., 0], 0.5), 'S_angle_median': q(S_arc[..., 1], 0.5), 'models': {}}
    if len(P):
        r_arc = compose(R_goal[:, None, :].expand(-1, len(th), -1), N_ax[:, None, :] * th[None, :, None])
        pp = P[:, None, :].expand(-1, len(th), -1)
        X = torch.cat([pp, r_arc, pp, R_goal[:, None, :].expand(-1, len(th), -1)], -1).reshape(-1, 12)
        seam = (r_arc.norm(dim=-1) * 360).reshape(len(P), -1)       # start's chart angle (deg)
        for m in MODELS:
            fn = get(m)
            Nr, tv, tw, tau = residual_at(fn, X, S_arc[..., 0].reshape(-1), S_arc[..., 1].reshape(-1))
            A = len(P)
            cur = {k: v.reshape(A, -1).numpy() for k, v in (('N', Nr), ('rot', tw), ('trans', tv), ('tau', tau))}
            cur['seam'] = seam.numpy()
            arc[m['label']] = cur
            medN = np.median(cur['N'], 0)
            # cap half-width: contiguous run around 180 where median N < 0.5
            i180 = int(np.argmin(np.abs(ARC_DEG - 180)))
            lo = hi = i180
            if medN[i180] < 0.5:
                while lo > 0 and medN[lo - 1] < 0.5:
                    lo -= 1
                while hi < len(ARC_DEG) - 1 and medN[hi + 1] < 0.5:
                    hi += 1
                cap = 0.5 * (ARC_DEG[hi] - ARC_DEG[lo]) + 2.5
            else:
                cap = 0.0
            summary['arc']['models'][m['label']] = {
                'N_at_180_median': float(medN[i180]), 'N_at_90_median': float(np.median(cur['N'][:, ARC_DEG == 90])),
                'rot_term_at_180_median': float(np.median(cur['rot'][:, i180])),
                'rot_term_at_90_median': float(np.median(cur['rot'][:, ARC_DEG == 90])),
                'tau_at_180_median': float(np.median(cur['tau'][:, i180])),
                'tau_at_90_median': float(np.median(cur['tau'][:, ARC_DEG == 90])),
                'cap_halfwidth_deg': float(cap),
                'N_lt_0p5_deg_per_turn': float((cur['N'] < 0.5).mean() * 360)}
            print('arc', m['label'], summary['arc']['models'][m['label']], flush=True)

# =========================================================================== pairs
pairs = {}
if 'pairs' not in SKIP:
    got = []
    tried = 0
    while sum(len(x) for x in got) < args.pairs and tried < 40 * args.pairs:
        n = 4096
        tried += n
        x0 = scene.sample_configs(n, gen).double()
        theta = torch.rand(n, 1, generator=gen, dtype=torch.float64) * math.pi
        r1 = compose(x0[:, 3:6], unit(n) * theta / TWO_PI)
        d = torch.randn(n, 3, generator=gen, dtype=torch.float64)
        d = d / d.norm(dim=1, keepdim=True) * 0.3 * torch.rand(n, 1, generator=gen, dtype=torch.float64)
        p1 = x0[:, :3] + d
        inb = (p1.abs() <= torch.as_tensor(scene.half_extent, dtype=torch.float64)).all(1)
        x1 = torch.cat([p1, r1], 1)
        both = torch.cat([x0, x1])
        S_d, S_a, free = speeds_of(both)
        ok = inb & free[:n] & free[n:]
        got.append(torch.cat([x0[ok], x1[ok], S_d[:n][ok, None].double(), S_a[:n][ok, None].double(),
                              S_d[n:][ok, None].double(), S_a[n:][ok, None].double()], 1))
    PR = torch.cat(got)[:args.pairs]
    print('pairs: {} free pairs from {} candidates'.format(len(PR), tried), flush=True)
    X = PR[:, :12]
    theta = geodesic_angle(X[:, 3:6], X[:, 9:12]).numpy() * 180 / math.pi
    seam0 = X[:, 3:6].norm(dim=1).numpy() * 360
    bins = np.arange(0, 181, 15)
    sbins = np.array([0, 60, 120, 150, 170, 181])
    summary['pairs'] = {'n': len(PR), 'theta_bins_deg': bins.tolist(), 'seam_bins_deg': sbins.tolist(), 'models': {}}
    for m in MODELS:
        fn = get(m)
        N0, tv0, tw0, tau = residual_at(fn, X, PR[:, 12], PR[:, 13], 0)
        N1, tv1, tw1, _ = residual_at(fn, X, PR[:, 14], PR[:, 15], 1)
        Nb = torch.cat([N0, N1]).numpy()
        rot = torch.cat([tw0, tw1]).numpy()
        th2 = np.concatenate([theta, theta])
        sm2 = np.concatenate([seam0, X[:, 9:12].norm(dim=1).numpy() * 360])
        cur = {'theta': th2, 'N': Nb, 'rot': rot, 'tau': np.concatenate([tau.numpy(), tau.numpy()]), 'seam': sm2}
        pairs[m['label']] = cur
        by_theta = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (th2 >= lo) & (th2 < hi)
            by_theta.append({'bin': [int(lo), int(hi)], 'n': int(s.sum()), 'N_median': q(Nb[s], 0.5),
                             'N_lt_0p5': float((Nb[s] < 0.5).mean()), 'rot_term_median': q(rot[s], 0.5),
                             'tau_median': q(cur['tau'][s], 0.5)})
        by_seam = []
        for lo, hi in zip(sbins[:-1], sbins[1:]):
            s = (sm2 >= lo) & (sm2 < hi) & (th2 < 120)
            by_seam.append({'bin': [int(lo), int(min(hi, 180))], 'n': int(s.sum()), 'N_median': q(Nb[s], 0.5) if s.any() else None,
                            'N_lt_0p5': float((Nb[s] < 0.5).mean()) if s.any() else None})
        summary['pairs']['models'][m['label']] = {'by_theta': by_theta, 'by_seam_theta_lt_120': by_seam,
                                                 'N_median_all': q(Nb, 0.5), 'loss_sqrtN_mean': float(((np.sqrt(Nb) - 1)**2).mean())}
        print('pairs', m['label'], 'N median by theta:', ' '.join('{:.2f}'.format(b['N_median']) for b in by_theta), flush=True)

# =========================================================================== mppi
mppi = {}
if 'mppi' not in SKIP:
    geo = geodesic_angle(test[:, 3:6], test[:, 9:12]).numpy() * 180 / math.pi
    big = np.nonzero(geo > args.big_deg)[0]
    summary['mppi'] = {'n_cases': int(len(big)), 'big_deg': args.big_deg, 'escape_deg': args.escape_deg, 'models': {}}
    print('mppi: {} test cases with relative rotation > {} deg'.format(len(big), args.big_deg), flush=True)

    def pose_dist(XP):
        dp = XP[:, DIM:DIM + 3] - XP[:, 0:3]
        ang = geodesic_angle(XP[:, 3:6].cpu(), XP[:, DIM + 3:DIM + 6].cpu()).to(XP.device).float()
        return torch.sqrt((dp**2).sum(1) + (ang / TWO_PI)**2), ang * 180 / math.pi, dp.norm(dim=1)

    def run(fn, XP, steps):
        B = XP.shape[0]
        dev = XP.device
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        angs, dps, steplen = [], [], []
        for it in range(steps):
            XP_tmp = XP.clone()[:, None, None, :].repeat(1, SAMPLE_NUM, HORIZON, 1)
            r = torch.full((B,), STEP, device=dev)[:, None, None, None]
            dP = STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, 1, DIM), dtype=torch.float32, device=dev) \
                + STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, HORIZON, DIM), dtype=torch.float32, device=dev)
            dP = dP / (torch.clamp(torch.norm(dP, dim=3, keepdim=True), min=r) / r)
            XP_tmp[..., 0:DIM] = XP_tmp[..., 0:DIM] + torch.cumsum(dP, dim=2)
            endpoints = XP_tmp[:, :, [0, -1], :]
            with torch.no_grad():
                cost = fn.TravelTimes(endpoints.reshape(-1, DIM * 2)).reshape(B, SAMPLE_NUM, 2)
            cost = 10 * cost[:, :, 0] + cost[:, :, 1]
            weight = torch.softmax(-TEMP * cost, dim=1)
            step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
            XP[:, 0:DIM] = XP[:, 0:DIM] + step_prior * (~done).unsqueeze(1)
            dis, ang, dpn = pose_dist(XP)
            done = done | (dis < 0.01)
            angs.append(ang.cpu().numpy()); dps.append(dpn.cpu().numpy())
            steplen.append((step_prior.norm(dim=1) / STEP).cpu().numpy())
            if bool(done.all()):
                break
        return np.stack(angs, 1), np.stack(dps, 1), np.stack(steplen, 1), done.cpu().numpy()

    for m in MODELS:
        fn = get(m)
        A, D, L, C = [], [], [], []
        for s in range(0, len(big), 125):
            XP = torch.tensor(test[big[s:s + 125]], dtype=torch.float32, device=args.device)
            a, d, l, c = run(fn, XP, args.steps)
            pad = args.steps - a.shape[1]
            A.append(np.pad(a, ((0, 0), (0, pad)), 'edge')); D.append(np.pad(d, ((0, 0), (0, pad)), 'edge'))
            L.append(np.pad(l, ((0, 0), (0, pad)), 'constant')); C.append(c)
        A, D, L, C = np.concatenate(A), np.concatenate(D), np.concatenate(L), np.concatenate(C)
        esc = np.array([int(np.argmax(a < args.escape_deg)) if (a < args.escape_deg).any() else args.steps for a in A])
        mppi[m['label']] = {'ang': A, 'dp': D, 'len': L, 'conv': C, 'escape': esc}
        summary['mppi']['models'][m['label']] = {
            'converged': float(C.mean()), 'escape_median': float(np.median(esc)),
            'never_escaped': float((esc >= args.steps).mean()),
            'final_angle_gt_big': float((A[:, -1] > args.big_deg).mean()),
            'step_frac_first20': float(L[:, :20].mean()), 'step_frac_first20_median': float(np.median(L[:, :20].mean(1))),
            'angle_after_20_median': float(np.median(A[:, 19])), 'angle_after_50_median': float(np.median(A[:, 49])),
            'dp_after_50_median': float(np.median(D[:, 49]))}
        print('mppi', m['label'], summary['mppi']['models'][m['label']], flush=True)


# =========================================================================== pooled report
def agg(vals):
    v = np.array([x for x in vals if x is not None], dtype=np.float64)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0, 'n': int(len(v))} if len(v) else None


def ms(a, scale=1.0, fmt='{:.2f}'):
    if a is None:
        return '   -   '
    return (fmt + ' +- ' + fmt).format(scale * a['mean'], scale * a['sd']) if a['n'] > 1 else fmt.format(scale * a['mean'])


summary['groups'] = {}
lines = ['antipode probe on {} ({} models, {} groups)'.format(args.dataPath, len(MODELS), len(GROUPS)), '']
if 'arc' in summary:
    lines += ['[arc] start rotated by theta about a body axis from a random goal rotation, free full turns, {} anchors'.format(summary['arc']['n_anchors']),
              '      N = Lie Eikonal residual at the start (1 = satisfied), rot = S_a |grad_w tau|; medians over anchors',
              '      cap = half-width (deg) of the band around theta = 180 where the median N < 0.5',
              '  {:<12} {:>2}  {:>12} {:>12}  {:>12} {:>12}  {:>12} {:>12}  {:>10}'.format(
                  'variant', 'n', 'N@90', 'N@180', 'rot@90', 'rot@180', 'tau@90', 'tau@180', 'cap deg')]
    for grp in GROUPS:
        S = [summary['arc']['models'][m['label']] for m in MODELS if m['group'] == grp and m['label'] in summary['arc']['models']]
        g = summary['groups'].setdefault(grp, {})
        for k in ('N_at_90_median', 'N_at_180_median', 'rot_term_at_90_median', 'rot_term_at_180_median',
                  'tau_at_90_median', 'tau_at_180_median', 'cap_halfwidth_deg'):
            g['arc_' + k] = agg([s[k] for s in S])
        lines.append('  {:<12} {:>2}  {:>12} {:>12}  {:>12} {:>12}  {:>12} {:>12}  {:>10}'.format(
            grp, len(S), ms(g['arc_N_at_90_median']), ms(g['arc_N_at_180_median']), ms(g['arc_rot_term_at_90_median']),
            ms(g['arc_rot_term_at_180_median']), ms(g['arc_tau_at_90_median'], fmt='{:.3f}'), ms(g['arc_tau_at_180_median'], fmt='{:.3f}'),
            ms(g['arc_cap_halfwidth_deg'], fmt='{:.0f}')))
if 'pairs' in summary:
    bins = summary['pairs']['theta_bins_deg']
    lines += ['', '[pairs] random free pairs ({}), Lie Eikonal residual N at both endpoints, median per relative-rotation bin (deg)'.format(summary['pairs']['n']),
              '  {:<12} '.format('variant') + ' '.join('{:>7}'.format('{}-{}'.format(lo, hi)) for lo, hi in zip(bins[:-1], bins[1:])) + '   | loss']
    for grp in GROUPS:
        S = [summary['pairs']['models'][m['label']] for m in MODELS if m['group'] == grp and m['label'] in summary['pairs']['models']]
        if not S:
            continue
        med = np.array([[b['N_median'] for b in s['by_theta']] for s in S]).mean(0)
        g = summary['groups'].setdefault(grp, {})
        g['pairs_N_median_by_theta'] = med.tolist()
        g['pairs_loss'] = agg([s['loss_sqrtN_mean'] for s in S])
        lines.append('  {:<12} '.format(grp) + ' '.join('{:7.2f}'.format(v) for v in med) + '   | {}'.format(ms(g['pairs_loss'], fmt='{:.4f}')))
    lines += ['', '        same, share of endpoints with N < 0.5']
    for grp in GROUPS:
        S = [summary['pairs']['models'][m['label']] for m in MODELS if m['group'] == grp and m['label'] in summary['pairs']['models']]
        if not S:
            continue
        fr = np.array([[b['N_lt_0p5'] for b in s['by_theta']] for s in S]).mean(0)
        summary['groups'][grp]['pairs_N_lt_0p5_by_theta'] = fr.tolist()
        lines.append('  {:<12} '.format(grp) + ' '.join('{:7.2f}'.format(v) for v in fr))
    sb = summary['pairs']['seam_bins_deg']
    lines += ['', '        pairs with relative rotation < 120 deg, N median by the endpoint\'s own chart angle |rotvec| (deg; 180 = seam)',
              '  {:<12} '.format('variant') + ' '.join('{:>8}'.format('{}-{}'.format(lo, min(hi, 180))) for lo, hi in zip(sb[:-1], sb[1:]))]
    for grp in GROUPS:
        S = [summary['pairs']['models'][m['label']] for m in MODELS if m['group'] == grp and m['label'] in summary['pairs']['models']]
        if not S:
            continue
        med = np.array([[b['N_median'] if b['N_median'] is not None else np.nan for b in s['by_seam_theta_lt_120']] for s in S]).mean(0)
        summary['groups'][grp]['pairs_N_median_by_seam'] = med.tolist()
        lines.append('  {:<12} '.format(grp) + ' '.join('{:8.2f}'.format(v) for v in med))
if 'mppi' in summary:
    lines += ['', '[mppi] forward MPPI (no collision check) on the {} test cases with relative rotation > {:.0f} deg, {} steps'.format(
        summary['mppi']['n_cases'], args.big_deg, args.steps),
        '       escape = steps until the geodesic angle to the goal first drops below {:.0f} deg; step = executed step / STEP, first 20 iterations'.format(args.escape_deg),
        '  {:<12} {:>2}  {:>13} {:>13} {:>13} {:>13} {:>13} {:>13}'.format(
            'variant', 'n', 'converged', 'escape med', 'never esc', 'step frac', 'ang@20', 'ang@50')]
    for grp in GROUPS:
        S = [summary['mppi']['models'][m['label']] for m in MODELS if m['group'] == grp and m['label'] in summary['mppi']['models']]
        if not S:
            continue
        g = summary['groups'].setdefault(grp, {})
        for k in ('converged', 'escape_median', 'never_escaped', 'step_frac_first20', 'angle_after_20_median', 'angle_after_50_median'):
            g['mppi_' + k] = agg([s[k] for s in S])
        lines.append('  {:<12} {:>2}  {:>13} {:>13} {:>13} {:>13} {:>13} {:>13}'.format(
            grp, len(S), ms(g['mppi_converged'], 100, '{:.1f}'), ms(g['mppi_escape_median'], fmt='{:.0f}'),
            ms(g['mppi_never_escaped'], 100, '{:.1f}'), ms(g['mppi_step_frac_first20']),
            ms(g['mppi_angle_after_20_median'], fmt='{:.0f}'), ms(g['mppi_angle_after_50_median'], fmt='{:.0f}')))
save_json(summary, os.path.join(args.out, 'summary.json'))
report = '\n'.join(lines)
print('\n' + report)
open(os.path.join(args.out, 'report.txt'), 'w').write(report + '\n')


# =========================================================================== figures
def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


gstyle = {grp: dict(color=COLORS[i % len(COLORS)], ls=('-', '--')[(i // len(COLORS)) % 2], lw=1.5) for i, grp in enumerate(GROUPS)}


def pooled(store, grp, key):
    return np.concatenate([store[m['label']][key] for m in MODELS if m['group'] == grp and m['label'] in store])


if arc:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, key, name in zip(axes, ('tau', 'rot', 'N'), ('tau(start, goal)', 'rotational term S_a |grad_w tau|', 'Eikonal residual N')):
        for grp in GROUPS:
            v = pooled(arc, grp, key)
            ax.plot(ARC_DEG, np.median(v, 0), label=grp, **gstyle[grp])
        ax.axvline(180, color=MUTED, lw=0.8)
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel('start rotated from the goal about a body axis (deg)', fontsize=8, color=MUTED)
        style(ax)
    axes[2].axhline(1.0, color=MUTED, lw=0.8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'arc.png'), dpi=150)
    plt.close(fig)
if pairs:
    bins = np.array(summary['pairs']['theta_bins_deg'])
    mid = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for grp in GROUPS:
        th = pooled(pairs, grp, 'theta'); Nb = pooled(pairs, grp, 'N'); rot = pooled(pairs, grp, 'rot'); tau = pooled(pairs, grp, 'tau')
        med = [np.median(Nb[(th >= lo) & (th < hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
        fr = [(Nb[(th >= lo) & (th < hi)] < 0.5).mean() for lo, hi in zip(bins[:-1], bins[1:])]
        tm = [np.median(tau[(th >= lo) & (th < hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
        axes[0].plot(mid, med, label=grp, **gstyle[grp]); axes[1].plot(mid, fr, **gstyle[grp]); axes[2].plot(mid, tm, **gstyle[grp])
    axes[0].axhline(1.0, color=MUTED, lw=0.8)
    for ax, t in zip(axes, ('median Eikonal residual N', 'share of endpoints with N < 0.5', 'median tau(start, goal)')):
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_xlabel('relative rotation start -> goal (deg)', fontsize=8, color=MUTED)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'pairs.png'), dpi=150)
    plt.close(fig)
if mppi:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for grp in GROUPS:
        A = pooled(mppi, grp, 'ang'); D = pooled(mppi, grp, 'dp'); L = pooled(mppi, grp, 'len')
        t = np.arange(1, A.shape[1] + 1)
        axes[0].plot(t, np.median(A, 0), label=grp, **gstyle[grp])
        axes[1].plot(t, np.median(D, 0), **gstyle[grp])
        axes[2].plot(t, np.median(L, 0), **gstyle[grp])
    for ax, t in zip(axes, ('geodesic angle to goal (deg, median)', 'position distance to goal (median)', 'executed step / STEP (median)')):
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_xlabel('MPPI iteration', fontsize=8, color=MUTED)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'mppi.png'), dpi=150)
    plt.close(fig)
print('wrote', args.out)
