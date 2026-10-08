"""Travel-time field analysis of one checkpoint (no planning).

    python analysis/field_analysis.py --kind lie --checkpoint <pt> --tag lie_s1 \
        --sets results/compare/sets --out results/compare/field/lie_s1.json --device cuda:0

Speeds are read off the gradient of tau(q, g) w.r.t. the configuration q, for
held-out q and generic goals g:
    S_t   = 1/|d tau/dp|                       translational speed
    S_rc  = 1/|d tau/dr|                       rotational speed, chart (rotvec) gradient
    S_rw  = 1/|Jr^{-T} d tau/dr|               rotational speed, body-frame (Lie) gradient
    S_iso = 1/|d tau/dq|                       isotropic 6-D speed
against the warped geometry targets each model was trained on
    S*_d (translational), S*_a (rotational), S*_c (metric_baseline's combined speed).

Sections of the output JSON:
  binned      speeds / targets / log-ratios / residuals on a translational-clearance x
              rotational-clearance grid, plus 1-D marginals and rotation-magnitude bins
  global      rank correlations with the targets, dynamic range, residuals
  goal_cv     variation of the speeds at a fixed q across 16 goals (a speed is a
              property of q; it should not depend on the goal)
  free_metric free-space pairs: which distance tau reproduces (Euclidean chart,
              block-L1 chart, SO(3)-geodesic, block-L1 geodesic)
  rot_pairs   pure rotations: tau vs geodesic angle vs chart distance, and 'wrap'
              pairs (close on SO(3), far apart in the chart) vs equal-angle controls
  cut_locus   the same pose written with two rotation vectors across the cut locus
  band_probe  on the clearance band, does each speed drop along the clearance-reducing
              directions (trans_n; rot_n as a chart line and as a Lie-group curve)
"""
import os
import sys
import math
import argparse
import time

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.append('.')
from analysis.common import (load_model, tau_and_grad, target_speeds, body_rot_grad,  # noqa: E402
                             geodesic_angle, save_json, TWO_PI, MARGIN)
from models import rnl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True, choices=['metric', 'shape', 'lie'])
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--sets', default='results/compare/sets')
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--K', type=int, default=4, help='goals per probe config')
ap.add_argument('--chunk', type=int, default=2048)
args = ap.parse_args()
torch.set_num_threads(8)
t_start = time.time()

model, fn = load_model(args.kind, args.checkpoint, args.device)
dev = args.device

import glob as _glob  # noqa: E402
_parts = [np.load(p) for p in sorted(_glob.glob(os.path.join(args.sets, 'probe_set*.npz')))]
S = {k: np.concatenate([p[k] for p in _parts]) for k in ('q', 'dist', 'angle', 'trans_n', 'rot_n', 'near')}
goals = torch.tensor(np.load(os.path.join(args.sets, 'goal_pool.npy')), dtype=torch.float32)
q = torch.tensor(S['q'], dtype=torch.float32)
dist = torch.tensor(S['dist'], dtype=torch.float64)
angle = torch.tensor(S['angle'], dtype=torch.float64)
N = len(q)
Sd, Sa, Sc = target_speeds(dist, angle)


def grads_at(qs, gs):
    """tau and per-block gradient quantities of tau(qs_i, gs_i) w.r.t. qs_i."""
    X = torch.cat([qs, gs], 1).to(dev)
    tau, dt = tau_and_grad(fn, X, chunk=args.chunk)
    g = dt[:, :6]
    gp, gr = g[:, :3], g[:, 3:6]
    gw = body_rot_grad(qs[:, 3:], gr)
    eps = 1e-12
    return {'tau': tau, 'gp': gp.norm(dim=1) + eps, 'gr': gr.norm(dim=1) + eps,
            'gw': gw.norm(dim=1) + eps, 'g': g.norm(dim=1) + eps}


# ---------------------------------------------------------------- per-config speeds (K goals)
K = args.K
gi = torch.arange(N * K) % len(goals)
rep = grads_at(q.repeat_interleave(K, 0), goals[gi])


def per_cfg_median(x):
    return x.view(N, K).median(dim=1).values


gp, gr, gw, gg = (per_cfg_median(rep[k]) for k in ('gp', 'gr', 'gw', 'g'))
St, Src, Srw, Siso = 1 / gp, 1 / gr, 1 / gw, 1 / gg
res = {
    'iso': Sc * gg,                                          # metric_baseline eq. (target 1)
    'blk_t': Sd * gp, 'blk_r': Sa * gr,                     # shape (blocked) eqs.
    'aniso': torch.sqrt(Sd ** 2 * gp ** 2 + Sa ** 2 * gw ** 2),  # lie (anisotropic) eq.
}
rot_share = (gr ** 2 / gg ** 2)                              # share of |grad|^2 in rotation

# Directional (finite-difference) speeds, model-agnostic: the time tau assigns to a
# short pure translation (h along a random unit u_p) or a short pure body-frame
# rotation (h turns about a random axis u_w), minus the network's self-distance
# tau(q, q).  S_dir = h / (tau(q, q + step) - tau(q, q)).
from analysis.common import so3_exp as _expd, so3_log as _logd  # noqa: E402
_gd = torch.Generator().manual_seed(11)
HD = 0.02
_up = torch.randn(N, 3, generator=_gd); _up = _up / _up.norm(dim=1, keepdim=True)
_uw = torch.randn(N, 3, generator=_gd); _uw = _uw / _uw.norm(dim=1, keepdim=True)
_qt = q.clone(); _qt[:, :3] += HD * _up
_qr = q.clone()
_qr[:, 3:] = (_logd(_expd(TWO_PI * q[:, 3:].double()) @ _expd(TWO_PI * HD * _uw.double())) / TWO_PI).float()
_self = tau_and_grad(fn, torch.cat([q, q], 1).to(dev), chunk=args.chunk, need_grad=False)[0]
_tt = tau_and_grad(fn, torch.cat([q, _qt], 1).to(dev), chunk=args.chunk, need_grad=False)[0]
_tr = tau_and_grad(fn, torch.cat([q, _qr], 1).to(dev), chunk=args.chunk, need_grad=False)[0]
Sdir_t = HD / (_tt - _self).clamp_min(1e-9)
Sdir_r = HD / (_tr - _self).clamp_min(1e-9)
# a 4x shorter step: a detour (move away, rotate where it is cheap, come back) has a
# cost that does not shrink with h while the in-place cost does, so short steps read
# the local speed; long steps read the local metric distance including detours.
HS = 0.005
_qt5 = q.clone(); _qt5[:, :3] += HS * _up
_qr5 = q.clone()
_qr5[:, 3:] = (_logd(_expd(TWO_PI * q[:, 3:].double()) @ _expd(TWO_PI * HS * _uw.double())) / TWO_PI).float()
_tt5 = tau_and_grad(fn, torch.cat([q, _qt5], 1).to(dev), chunk=args.chunk, need_grad=False)[0]
_tr5 = tau_and_grad(fn, torch.cat([q, _qr5], 1).to(dev), chunk=args.chunk, need_grad=False)[0]
Sdir_t5 = HS / (_tt5 - _self).clamp_min(1e-9)
Sdir_r5 = HS / (_tr5 - _self).clamp_min(1e-9)
theta = (q[:, 3:].double().norm(dim=1) * 360.0)              # rotation angle of q, degrees

D_EDGES = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0, np.inf]             # d / margin
A_EDGES = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 181.0]     # rotational clearance, deg
T_EDGES = [0.0, 45.0, 90.0, 135.0, 180.1]                    # rotation magnitude, deg
dbin = np.digitize((dist / MARGIN).numpy(), D_EDGES[1:-1])
abin = np.digitize(np.degrees(angle.numpy()), A_EDGES[1:-1])
tbin = np.digitize(theta.numpy(), T_EDGES[1:-1])

lr = {'t': torch.log(St / Sd), 'rc': torch.log(Src / Sa), 'rw': torch.log(Srw / Sa),
      'iso': torch.log(Siso / Sc), 'dir_t': torch.log(Sdir_t / Sd), 'dir_r': torch.log(Sdir_r / Sa),
      'dir5_t': torch.log(Sdir_t5 / Sd), 'dir5_r': torch.log(Sdir_r5 / Sa)}


def med(x, m):
    v = x[m]
    return float(v.median()) if v.numel() else None


def bin_stats(m):
    m = torch.as_tensor(m)
    n = int(m.sum())
    out = {'n': n}
    if n == 0:
        return out
    for k, v in (('S_t', St), ('S_rc', Src), ('S_rw', Srw), ('S_iso', Siso),
                 ('Sdir_t', Sdir_t), ('Sdir_r', Sdir_r), ('Sdir5_t', Sdir_t5), ('Sdir5_r', Sdir_r5),
                 ('Sstar_d', Sd), ('Sstar_a', Sa), ('Sstar_c', Sc), ('rot_share', rot_share)):
        out[k] = med(v, m)
    for k, v in lr.items():
        out['logratio_' + k] = med(v, m)
        out['abslog_' + k] = med(v.abs(), m)
    for k, v in res.items():
        out['res_' + k] = med(v, m)
        out['absres_' + k] = med((v - 1).abs(), m)
    return out


binned = {'grid': {}, 'dist': {}, 'angle': {}, 'theta': {}}
for i in range(len(D_EDGES) - 1):
    binned['dist'][str(i)] = bin_stats(dbin == i)
    for j in range(len(A_EDGES) - 1):
        binned['grid']['{}_{}'.format(i, j)] = bin_stats((dbin == i) & (abin == j))
for j in range(len(A_EDGES) - 1):
    binned['angle'][str(j)] = bin_stats(abin == j)
for t in range(len(T_EDGES) - 1):
    binned['theta'][str(t)] = bin_stats(tbin == t)
binned['edges'] = {'dist_over_margin': D_EDGES, 'angle_deg': A_EDGES, 'theta_deg': T_EDGES}


def spear(a, b):
    return float(spearmanr(a.numpy(), b.numpy()).correlation)


tight = ((dist / MARGIN) < 0.2) | (np.degrees(angle) < 5)
free = ((dist / MARGIN) >= 1.0) & (np.degrees(angle) >= 30)
glob = {
    'n_probe': N,
    'spearman': {'Sdir_t~Sstar_d': spear(Sdir_t, Sd), 'Sdir_r~Sstar_a': spear(Sdir_r, Sa),
                 'Sdir5_t~Sstar_d': spear(Sdir_t5, Sd), 'Sdir5_r~Sstar_a': spear(Sdir_r5, Sa),
                 'S_t~Sstar_d': spear(St, Sd), 'S_rc~Sstar_a': spear(Src, Sa),
                 'S_rw~Sstar_a': spear(Srw, Sa), 'S_iso~Sstar_c': spear(Siso, Sc),
                 'S_iso~min(Sd,Sa)': spear(Siso, torch.minimum(Sd, Sa))},
    'dynamic_range': {k: float(v[free].median() / v[tight].median())
                      for k, v in (('S_t', St), ('S_rc', Src), ('S_rw', Srw), ('S_iso', Siso),
                                   ('Sdir_t', Sdir_t), ('Sdir_r', Sdir_r), ('Sdir5_t', Sdir_t5), ('Sdir5_r', Sdir_r5),
                                   ('Sstar_d', Sd), ('Sstar_a', Sa))},
    'self_tau_median': float(_self.median()),
    'median_abslog': {k: float(v.abs().median()) for k, v in lr.items()},
    'median_absres': {k: float((v - 1).abs().median()) for k, v in res.items()},
    'theta_trend_logratio_rc_vs_rw': {
        str(t): [med(lr['rc'], torch.as_tensor(tbin == t)), med(lr['rw'], torch.as_tensor(tbin == t))]
        for t in range(len(T_EDGES) - 1)},
}

# ---------------------------------------------------------------- goal consistency
M = min(2000, N)
sel = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:M]
G = len(goals)
gc = grads_at(q[sel].repeat_interleave(G, 0), goals.repeat(M, 1))


def cv(x):
    x = (1 / x).view(M, G)
    return float((x.std(dim=1) / x.mean(dim=1)).median())


goal_cv = {'S_t': cv(gc['gp']), 'S_rc': cv(gc['gr']), 'S_rw': cv(gc['gw']), 'S_iso': cv(gc['g'])}

def fit(y, x):
    a = float((x * y).sum() / (x * x).sum())
    ss_res = float(((y - a * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {'slope': a, 'R2': 1 - ss_res / ss_tot, 'spearman': spear(y, x),
            'median_rel_err': float(((y - a * x).abs() / (a * x)).median())}


# ---------------------------------------------------------------- free-space metric identification
free_metric = None
if os.path.exists(os.path.join(args.sets, 'free_pairs.npy')):
    FP = torch.tensor(np.load(os.path.join(args.sets, 'free_pairs.npy')), dtype=torch.float32)
    q0, q1 = FP[:, :6], FP[:, 6:]
    fr = grads_at(q0, q1)
    dp = (q1[:, :3] - q0[:, :3]).double().norm(dim=1)
    drc = (q1[:, 3:] - q0[:, 3:]).double().norm(dim=1)
    dgeo = geodesic_angle(q0[:, 3:], q1[:, 3:]) / TWO_PI
    cands = {'euclid_chart': torch.sqrt(dp ** 2 + drc ** 2), 'L1_chart': dp + drc,
             'euclid_geodesic': torch.sqrt(dp ** 2 + dgeo ** 2), 'L1_geodesic': dp + dgeo}
    tau_f = fr['tau']



    free_metric = {k: fit(tau_f, v) for k, v in cands.items()}
    free_metric['best'] = max(cands, key=lambda k: free_metric[k]['R2'])
    free_metric['grad_norms_at_q0'] = {
        '|g|': float(fr['g'].median()), '|g_p|': float(fr['gp'].median()),
        '|g_r|': float(fr['gr'].median()), '|g_w|': float(fr['gw'].median()),
        'sqrt(|g_p|^2+|g_w|^2)': float(torch.sqrt(fr['gp'] ** 2 + fr['gw'] ** 2).median())}
    free_metric['n'] = len(FP)

# ---------------------------------------------------------------- local metric identification
# From probe points with some clearance, step h along mixed directions
#   phi = 0 (pure translation) ... 90 deg (pure body-frame rotation):
#   q1 = (p + h cos(phi) u_p,  R Exp(2 pi h sin(phi) u_w))
# (a) effective norm exponent p from tau(0), tau(45), tau(90):
#       sqrt(2) tau(45) = (tau(0)^p + tau(90)^p)^(1/p)
#     p = 1: translation and rotation times add (block-L1, the blocked Eikonal's
#     free-space solution); p = 2: they combine in quadrature (Euclidean /
#     Riemannian: the isotropic and anisotropic Eikonals).
# (b) dual norms at q of the start gradient, per phi: x = S*_d |g_p|, y = S*_a |g_w|
#     (blocked predicts x = y = 1 at every phi; anisotropic predicts x^2 + y^2 = 1,
#     moving from (1, 0) to (0, 1); isotropic predicts S*_c |g| = 1).
gen_lm = torch.Generator().manual_seed(5)
cand = torch.nonzero((dist >= 0.4 * MARGIN) & (angle >= math.radians(10))).flatten()
cand = cand[torch.randperm(len(cand), generator=gen_lm)[:1000]]
lq = q[cand]
nl = len(lq)
up = torch.randn(nl, 3, generator=gen_lm); up = up / up.norm(dim=1, keepdim=True)
uw = torch.randn(nl, 3, generator=gen_lm); uw = uw / uw.norm(dim=1, keepdim=True)
H = 0.05
PHIS = [0.0, 22.5, 45.0, 67.5, 90.0]
from analysis.common import so3_exp as _exp, so3_log as _log  # noqa: E402
Rq = _exp(TWO_PI * lq[:, 3:].double())
lm = {'n': nl, 'h': H, 'phi_deg': PHIS, 'per_phi': {}}
tau_phi = {}
self_tau = grads_at(lq, lq)['tau']
for ph in PHIS:
    a, b = H * math.cos(math.radians(ph)), H * math.sin(math.radians(ph))
    q1 = lq.clone()
    q1[:, :3] = lq[:, :3] + a * up
    q1[:, 3:] = (_log(Rq @ _exp(TWO_PI * b * uw.double())) / TWO_PI).float()
    gq = grads_at(lq, q1)
    tau_phi[ph] = (gq['tau'] - self_tau).clamp_min(1e-9)
    sd_, sa_, sc_ = Sd[cand], Sa[cand], Sc[cand]
    x = sd_ * gq['gp']; y = sa_ * gq['gw']; yc = sa_ * gq['gr']
    lm['per_phi'][str(ph)] = {
        'x=Sd|g_p|': float(x.median()), 'y=Sa|g_w|': float(y.median()), 'yc=Sa|g_r|': float(yc.median()),
        'x2+y2': float((x ** 2 + y ** 2).median()), 'iso=Sc|g|': float((sc_ * gq['g']).median()),
        'tau_minus_self': float(tau_phi[ph].median())}
t0_, t45, t90 = tau_phi[0.0], tau_phi[45.0], tau_phi[90.0]
target = math.sqrt(2) * t45
lo = torch.full_like(t0_, 0.25); hi = torch.full_like(t0_, 16.0)
for _ in range(60):                       # bisection: (t0^p + t90^p)^(1/p) decreases in p
    mid = 0.5 * (lo + hi)
    val = (t0_ ** mid + t90 ** mid) ** (1 / mid)
    big = val > target
    lo = torch.where(big, mid, lo); hi = torch.where(big, hi, mid)
p_est = 0.5 * (lo + hi)
lm['p_exponent'] = {'median': float(p_est.median()), 'q25': float(p_est.quantile(0.25)),
                    'q75': float(p_est.quantile(0.75)),
                    'frac_p_lt_1p5': float((p_est < 1.5).double().mean()),
                    'frac_p_gt_1p5': float((p_est >= 1.5).double().mean())}
lm['additivity_ratio_median'] = float((target / (t0_ + t90)).median())   # 1 = L1, 1/sqrt(2)*sqrt2.. = Euclid

# ---------------------------------------------------------------- rotation-only pairs
rot_pairs = None
if os.path.exists(os.path.join(args.sets, 'rot_pairs.npz')):
    R = np.load(os.path.join(args.sets, 'rot_pairs.npz'))
    kind = R['kind']
    rq0 = torch.tensor(R['q0'], dtype=torch.float32)
    rq1 = torch.tensor(R['q1'], dtype=torch.float32)
    rr = grads_at(rq0, rq1)
    geo_t = torch.tensor(R['geo'], dtype=torch.float64) / TWO_PI
    chart = (rq1[:, 3:] - rq0[:, 3:]).double().norm(dim=1)
    gen_m = torch.as_tensor(kind == 'generic')
    wrap_m = torch.as_tensor(kind == 'wrap')
    ctrl_m = torch.as_tensor(kind == 'control')
    # wrap and control pairs are aligned row by row (same position, start and axis)
    wrap_ratio = (rr['tau'][wrap_m] / rr['tau'][ctrl_m].clamp_min(1e-9))
    rot_pairs = {
        'paired_wrap_over_control_median': float(wrap_ratio.median()),
        'paired_wrap_over_control_iqr': [float(wrap_ratio.quantile(0.25)), float(wrap_ratio.quantile(0.75))],
        'chart_ratio_wrap_over_control_median': float((chart[wrap_m] / chart[ctrl_m]).median()),
        'generic_fit_geodesic': fit(rr['tau'][gen_m], geo_t[gen_m]),
        'generic_fit_chart': fit(rr['tau'][gen_m], chart[gen_m]),
        'wrap_over_control_tau': float(rr['tau'][wrap_m].median() / rr['tau'][ctrl_m].median()),
        'tau_median': {'wrap': float(rr['tau'][wrap_m].median()), 'control': float(rr['tau'][ctrl_m].median())},
        'geodesic_turns_median': {'wrap': float(geo_t[wrap_m].median()), 'control': float(geo_t[ctrl_m].median())},
        'chart_dist_median': {'wrap': float(chart[wrap_m].median()), 'control': float(chart[ctrl_m].median())},
        # free-space rotational slowness for pure rotations (S* = 1 there)
        'grad_norms_generic': {'|g_r|': float(rr['gr'][gen_m].median()), '|g_w|': float(rr['gw'][gen_m].median()),
                               '|g_p|': float(rr['gp'][gen_m].median())},
    }
    # tau vs geodesic angle, binned, for generic pairs (chart-distance and tau medians)
    gb = np.digitize(np.degrees(R['geo'][kind == 'generic']), [30, 60, 90, 120, 150])
    rot_pairs['generic_by_geodesic_bin'] = {
        str(b): {'n': int((gb == b).sum()),
                 'tau': float(rr['tau'][gen_m][torch.as_tensor(gb == b)].median()) if (gb == b).any() else None,
                 'geo_turns': float(geo_t[gen_m][torch.as_tensor(gb == b)].median()) if (gb == b).any() else None,
                 'chart': float(chart[gen_m][torch.as_tensor(gb == b)].median()) if (gb == b).any() else None}
        for b in range(6)}

# ---------------------------------------------------------------- cut-locus consistency
cut_locus = None
if os.path.exists(os.path.join(args.sets, 'cut_set.npz')):
    C = np.load(os.path.join(args.sets, 'cut_set.npz'))
    cq = torch.tensor(C['q'], dtype=torch.float32)
    cqa = torch.tensor(C['q_alt'], dtype=torch.float32)
    nc = len(cq)
    GC = min(8, len(goals))
    cg = goals[:GC].repeat(nc, 1)
    a1 = grads_at(cq.repeat_interleave(GC, 0), cg)
    a2 = grads_at(cqa.repeat_interleave(GC, 0), cg)
    rel_tau = (a1['tau'] - a2['tau']).abs() / a1['tau'].abs().clamp_min(1e-6)
    cut_locus = {
        'n': nc,
        'median_rel_tau_diff': float(rel_tau.median()),
        'p90_rel_tau_diff': float(rel_tau.quantile(0.9)),
        'median_log_speed_ratio': {
            'S_t': float(torch.log(a1['gp'] / a2['gp']).abs().median()),
            'S_rw': float(torch.log(a1['gw'] / a2['gw']).abs().median()),
            'S_iso': float(torch.log(a1['g'] / a2['g']).abs().median())},
    }

# ---------------------------------------------------------------- band probes
band = (dist < MARGIN)
bq = q[band]
tn = torch.tensor(S['trans_n'], dtype=torch.float32)[band]
rn = torch.tensor(S['rot_n'], dtype=torch.float32)[band]
nb = len(bq)
gbq = goals[torch.arange(nb) % len(goals)]
base = grads_at(bq, gbq)
band_probe = {'n': nb}
for mode in ('orig', 'lie'):
    qt, qr, valid = rnl.make_probes(bq, tn, rn, 0.01, mode)
    pt = grads_at(qt, gbq)
    pr = grads_at(qr, gbq)
    v = valid.cpu()
    band_probe[mode] = {
        'trans_probe_frac_drop': {'S_t': float((pt['gp'] > base['gp']).double().mean()),
                                  'S_iso': float((pt['g'] > base['g']).double().mean())},
        'rot_probe_frac_drop': {'S_rc': float((pr['gr'] > base['gr'])[v].double().mean()),
                                'S_rw': float((pr['gw'] > base['gw'])[v].double().mean()),
                                'S_iso': float((pr['g'] > base['g'])[v].double().mean())},
    }

out = {'tag': args.tag, 'kind': args.kind, 'checkpoint': args.checkpoint,
       'binned': binned, 'global': glob, 'goal_cv': goal_cv, 'free_metric': free_metric,
       'local_metric': lm, 'rot_pairs': rot_pairs, 'cut_locus': cut_locus, 'band_probe': band_probe,
       'seconds': time.time() - t_start}
save_json(out, args.out)
print('wrote', args.out, '({:.0f}s)'.format(time.time() - t_start))
print('global:', glob['spearman'], glob['median_abslog'])
if free_metric:
    print('free_metric best:', free_metric['best'], {k: round(v['R2'], 4) for k, v in free_metric.items() if isinstance(v, dict) and 'R2' in v})
print('local metric p:', lm['p_exponent'], 'additivity', lm['additivity_ratio_median'])
if rot_pairs:
    print('rot wrap/control (paired):', rot_pairs['paired_wrap_over_control_median'])
if cut_locus:
    print('cut rel diff:', cut_locus['median_rel_tau_diff'])
