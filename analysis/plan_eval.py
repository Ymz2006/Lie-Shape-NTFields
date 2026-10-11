"""Planning evaluation with per-case path metrics.

    python analysis/plan_eval.py --kind lie --checkpoint <pt> \
        --dataPath testing_data/3dshape/Tshape3d_env1_tight_3d --out <json> --device cuda:0

The planner, endpoint filter and collision test are copied from
evaluate_training_3d_batched.py at its defaults (forward MPPI: 50 samples,
horizon 5, step 0.015, temperature 50, momentum 0, 200 iterations, convergence
ball 0.01; collision = an environment surface point inside the placed shape mesh
at any waypoint).  Its 'regular/forward' success rate is reproduced, and per case
this also records:
    iters, path translation length, path rotation length (chart |dr| and true SO(3)
    geodesic, in turns), their ratios to the straight start-goal values, minimum
    translational clearance along the path (signed distance, capped at 0.05),
    tau(start, goal), and how often tau(waypoint, goal) increases along the path.
"""
import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch
import igl
from scipy.spatial import cKDTree

sys.path.append('.')
from analysis.common import load_model, geodesic_angle, save_json, TWO_PI  # noqa: E402
from dataprocessing.preprocess_obj import load_obj, sample_surface_points, _rotvec_to_matrix_np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True, choices=['metric', 'shape', 'lie', 'twist', 'lie_rnl', 'rhombus', 'lie_iso'])
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--dataPath', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--cases', type=int, default=0, help='0 = all')
ap.add_argument('--batch', type=int, default=250)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--group-conv', action='store_true',
                help='Converge on SE(3) instead of in the chart: |dp|^2 + (geodesic angle / 2pi)^2 '
                     '< 0.01^2, so reaching the goal pose through the rotation-vector cut locus '
                     'counts.  MPPI dynamics are unchanged.')
ap.add_argument('--planner', default='mppi', choices=['mppi', 'twist', 'mppi_twist', 'mppi_lie'],
                help='mppi: the evaluator\'s forward MPPI.  twist: the optimal-control law of the '
                     'twist derivation, xi* = -G0^{-1} p_b / |p_b|_{G0^{-1}} (p_b = left-trivialized '
                     'gradient of tau(., goal); G0 = the checkpoint\'s twist metric, I otherwise), '
                     'integrated as g <- g exp(h xi*).  mppi_twist: MPPI whose samples are body '
                     'twists from the admissible ellipsoid xi^T G0 xi <= STEP^2 (xi ~ N(0, STEP^2 G0^{-1}), '
                     'G0-length capped at STEP), applied as g exp(xi). '
                     'mppi_lie: unit body tangents, exact Riemannian exponential, '
                     'common t=STEP and normalized product-geodesic convergence.')
ap.add_argument('--twist-step', type=float, default=0.0075, help='twist planner: G0-length h per step')
ap.add_argument('--twist-steps', type=int, default=600, help='twist planner: max steps')
ap.add_argument('--temp', type=float, default=50.0,
                help='MPPI temperature (softmax(-temp * cost)); scaling tau by c is the same as temp x c.')
args = ap.parse_args()
torch.set_num_threads(4)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

DIM = 6
STEP, TEMP, SAMPLE_NUM, HORIZON = 0.015, args.temp, 50, 5
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
CLEAR_CAP = 0.03

meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))


def _resolve(path):
    return path if os.path.exists(path) else os.path.join('./datasets/3dshape', os.path.basename(path))


env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)
V_sh, F_sh, _ = load_obj(_resolve(meta['shape_obj']))
shape_V = np.ascontiguousarray((V_sh - 0.5 * (V_sh.min(0) + V_sh.max(0))) / env_scale * float(meta['shape_scale']), dtype=np.float64)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())
V_env, F_env, _ = load_obj(_resolve(meta['env_obj']))
env_pts = np.ascontiguousarray(sample_surface_points((V_env - env_center) / env_scale, F_env, 50000), dtype=np.float64)
env_tree = cKDTree(env_pts)


def path_clearance(P):
    """Per-waypoint clearance of the placed shape along a path P (T, 6): the minimum
    signed distance (>0 outside) of the env points near the shape, capped at
    CLEAR_CAP.  Negative = collision -- the same decision as the evaluator's
    check_trajectory_collision (env points within shape_radius, any negative
    signed distance); points out to shape_radius + CLEAR_CAP only add positive
    distances."""
    P = np.asarray(P, dtype=np.float64)
    out = np.full(len(P), CLEAR_CAP)
    for k, sel in enumerate(env_tree.query_ball_point(P[:, :3], r=shape_radius + CLEAR_CAP)):
        if not sel:
            continue
        R = _rotvec_to_matrix_np(P[k, 3:6] * TWO_PI)
        S = igl.signed_distance(np.ascontiguousarray((env_pts[sel] - P[k, :3]) @ R),
                                shape_V, shape_F, _SDF_SIGN)[0]
        out[k] = min(float(S.min()), CLEAR_CAP)
    return out


def collides(cfg):
    return bool(path_clearance(np.asarray(cfg)[None])[0] < 0.0)


# Clearance pass runs in worker processes (forked before CUDA is initialised).
import multiprocessing as mp  # noqa: E402
POOL = mp.get_context('fork').Pool(args.workers) if args.workers > 1 else None
model, fn = load_model(args.kind, args.checkpoint, args.device)


def wrap_turns(r):
    """Canonical rotation vector (|theta| <= pi) of r (B,3), stored in turns (rotvec/2pi)."""
    n = r.norm(dim=-1, keepdim=True)
    nw = torch.remainder(n + 0.5, 1.0) - 0.5
    return torch.where(n > 0.5, r / n.clamp(min=1e-12) * nw, r)


def pose_dist(XP):
    """Distance from the start slice of XP (B, 12) to its goal slice: the chart
    norm, or with --group-conv the SE(3) distance sqrt(|dp|^2 + (theta/2pi)^2) with
    theta the geodesic angle between the two rotations (any rotation vector)."""
    if not args.group_conv and args.planner == 'mppi':
        return torch.norm(XP[:, DIM:2 * DIM] - XP[:, 0:DIM], dim=1)
    dp = XP[:, DIM:DIM + 3] - XP[:, 0:3]
    q = []
    for rv in (XP[:, 3:6], XP[:, DIM + 3:DIM + 6]):
        phi = TWO_PI * rv
        th = phi.norm(dim=1, keepdim=True)
        q.append(torch.cat([torch.cos(th / 2), 0.5 * torch.sinc(th / TWO_PI) * phi], 1))
    c = (q[0] * q[1]).sum(1).abs().clamp(max=1.0)
    theta = 2.0 * torch.arccos(c)
    return torch.sqrt((dp**2).sum(1) + (theta / TWO_PI)**2)


def MPPI_batched(XP, steps):
    """Verbatim logic of evaluate_training_3d_batched.MPPI_batched at its defaults."""
    B = XP.shape[0]
    dev = XP.device
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    recorded = [XP[:, 0:DIM].clone()]
    for it in range(steps):
        XP_tmp = XP.clone()[:, None, None, :].repeat(1, SAMPLE_NUM, HORIZON, 1)
        r = torch.full((B,), STEP, device=dev)[:, None, None, None]
        dP = STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, 1, DIM), dtype=torch.float32, device=dev) \
            + STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, HORIZON, DIM), dtype=torch.float32, device=dev)
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=r) / r)
        XP_tmp[..., 0:DIM] = XP_tmp[..., 0:DIM] + torch.cumsum(dP, dim=2)
        endpoints = XP_tmp[:, :, [0, -1], :]
        with torch.no_grad():
            cost = fn.TravelTimes(endpoints.reshape(-1, DIM * 2)).reshape(B, SAMPLE_NUM, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        weight = torch.softmax(-TEMP * cost, dim=1)
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        XP[:, 0:DIM] = XP[:, 0:DIM] + step_prior * (~done).unsqueeze(1)
        dis = pose_dist(XP)
        recorded.append(XP[:, 0:DIM].clone())
        newly = (dis < 0.01) & (~done)
        conv_step[newly] = it
        done = done | (dis < 0.01)
        if bool(done.all()):
            break
    success = done.cpu().numpy()
    cs = conv_step.cpu().numpy()
    goal = XP[:, DIM:2 * DIM]
    paths = []
    for b in range(B):
        end = int(cs[b]) + 2 if success[b] else len(recorded)
        p = torch.cat([recorded[k][b:b + 1] for k in range(end)] + [goal[b:b + 1]]).cpu().numpy()
        paths.append(p)
    return paths, cs.tolist(), success.tolist()


def MPPI_twist(XP, steps):
    """MPPI_batched with samples drawn as body twists from the control set of the
    twist derivation: xi = STEP * L (z1 + z2_h), L L^T = G0^{-1}, each horizon step's
    G0-length capped at STEP (the chart-norm cap of MPPI_batched when G0 = I).  Twists
    map to chart increments at the current pose, dp = R v, dr = Jr^{-1}(r) w, and are
    accumulated along the horizon as in MPPI_batched; sampled endpoints and the
    executed state are wrapped to canonical rotation vectors (same poses), so
    convergence is always measured on SE(3)."""
    from models.lie_twist.model_function_metric import so3_exp, so3_right_jacobian_inv
    G0 = np.asarray(getattr(model, 'G0', np.eye(6)), dtype=np.float64)
    L = torch.as_tensor(np.linalg.cholesky(np.linalg.inv(G0)), dtype=torch.float32, device=XP.device)
    G0t = torch.as_tensor(G0, dtype=torch.float32, device=XP.device)
    B = XP.shape[0]
    dev = XP.device
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    recorded = [XP[:, 0:DIM].clone()]
    for it in range(steps):
        z = STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, 1, DIM), device=dev) \
            + STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, HORIZON, DIM), device=dev)
        xi = z @ L.T
        gnorm = torch.sqrt(torch.einsum('...i,ij,...j->...', xi, G0t, xi))[..., None]
        xi = xi / (torch.clamp(gnorm, min=STEP) / STEP)
        R = so3_exp(TWO_PI * XP[:, 3:6])
        Jri = so3_right_jacobian_inv(TWO_PI * XP[:, 3:6])
        dP = torch.cat([torch.einsum('bij,bshj->bshi', R, xi[..., :3]),
                        torch.einsum('bij,bshj->bshi', Jri, xi[..., 3:])], -1)
        XP_tmp = XP.clone()[:, None, None, :].repeat(1, SAMPLE_NUM, HORIZON, 1)
        XP_tmp[..., 0:DIM] = XP_tmp[..., 0:DIM] + torch.cumsum(dP, dim=2)
        endpoints = XP_tmp[:, :, [0, -1], :].reshape(-1, DIM * 2)
        endpoints[:, 3:6] = wrap_turns(endpoints[:, 3:6])
        with torch.no_grad():
            cost = fn.TravelTimes(endpoints).reshape(B, SAMPLE_NUM, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        weight = torch.softmax(-TEMP * cost, dim=1)
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        XP[:, 0:DIM] = XP[:, 0:DIM] + step_prior * (~done).unsqueeze(1)
        XP[:, 3:6] = wrap_turns(XP[:, 3:6])
        dis = pose_dist(XP)
        recorded.append(XP[:, 0:DIM].clone())
        newly = (dis < 0.01) & (~done)
        conv_step[newly] = it
        done = done | (dis < 0.01)
        if bool(done.all()):
            break
    success = done.cpu().numpy()
    cs = conv_step.cpu().numpy()
    goal = XP[:, DIM:2 * DIM]
    paths = []
    for b in range(B):
        end = int(cs[b]) + 2 if success[b] else len(recorded)
        p = torch.cat([recorded[k][b:b + 1] for k in range(end)] + [goal[b:b + 1]]).cpu().numpy()
        paths.append(p)
    return paths, cs.tolist(), success.tolist()


def twist_descent(XP, steps):
    """Optimal feedback control from the twist derivation: at the current pose,
    xi* = -G0^{-1} p_b / sqrt(p_b^T G0^{-1} p_b), followed for a G0-length h.  This is
    exactly time-optimal when the speed is a scalar S(g) and G0 is constant (the
    'dist' / 'combined' speed forms); for the block form it is the free-space
    direction.  First-order chart update p += h R v, r += Jr^{-1}(r) h w (w per turn),
    then r is wrapped to its canonical rotation vector (same pose).  An episode stops when it converges, or when tau(., goal) has not
    dropped by 1e-4 over the last 50 steps (stuck)."""
    from models.lie_twist.model_function_metric import body_twist_grad, so3_exp, so3_right_jacobian_inv
    G0 = np.asarray(getattr(model, 'G0', np.eye(6)), dtype=np.float64)
    G0i = torch.as_tensor(np.linalg.inv(G0), dtype=torch.float32, device=XP.device)
    h = args.twist_step
    B = XP.shape[0]
    dev = XP.device
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    stuck = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    recorded = [XP[:, 0:DIM].clone()]
    tau_hist = []
    for it in range(steps):
        tau, _, X = fn.network.out(XP)
        d = torch.autograd.grad(tau.sum(), X)[0][:, :DIM]
        pb = body_twist_grad(X[:, :DIM].detach(), d).detach()
        Gp = pb @ G0i.T
        xi = -Gp / torch.sqrt((pb * Gp).sum(1, keepdim=True) + 1e-12)
        active = (~done) & (~stuck)
        with torch.no_grad():
            r = XP[:, 3:6]
            R = so3_exp(TWO_PI * r)
            dp = h * torch.einsum('bij,bj->bi', R, xi[:, :3])
            dr = torch.einsum('bij,bj->bi', so3_right_jacobian_inv(TWO_PI * r), h * xi[:, 3:])
            XP[:, 0:3] = XP[:, 0:3] + dp * active[:, None]
            XP[:, 3:6] = wrap_turns(XP[:, 3:6] + dr * active[:, None])
            dis = pose_dist(XP)
            recorded.append(XP[:, 0:DIM].clone())
            newly = (dis < 0.01) & (~done)
            conv_step[newly] = it
            done = done | (dis < 0.01)
            tau_hist.append(tau[:, 0].detach())
            if it >= 50:
                stuck = stuck | ((~done) & (tau_hist[-51] - tau_hist[-1] < 1e-4))
        if bool((done | stuck).all()):
            break
    success = done.cpu().numpy()
    cs = conv_step.cpu().numpy()
    goal = XP[:, DIM:2 * DIM]
    paths = []
    for b in range(B):
        end = int(cs[b]) + 2 if success[b] else len(recorded)
        p = torch.cat([recorded[k][b:b + 1] for k in range(end)] + [goal[b:b + 1]]).cpu().numpy()
        # drop the frozen tail of a stuck episode (repeated identical waypoints)
        keep = np.concatenate([[True], np.abs(np.diff(p[:-1], axis=0)).sum(1) > 0, [True]])
        paths.append(p[keep])
    return paths, cs.tolist(), success.tolist()


arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float32)
n = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
arr = arr[:n]

t0 = time.time()
ends = [arr[i].reshape(2, 6) for i in range(n)]
end_clear = POOL.map(path_clearance, ends) if POOL is not None else [path_clearance(e) for e in ends]
ok_idx = [i for i in range(n) if (end_clear[i] >= 0.0).all()]
t_filter = time.time() - t0

records = []
t_plan = t_check = 0.0
for s in range(0, len(ok_idx), args.batch):
    idx = ok_idx[s:s + args.batch]
    XP = torch.tensor(arr[idx]).to(args.device)
    ta = time.time()
    if args.planner == 'twist':
        paths, iters, succ = twist_descent(XP.clone(), args.twist_steps)
    elif args.planner == 'mppi_twist':
        paths, iters, succ = MPPI_twist(XP.clone(), args.steps)
    elif args.planner == 'mppi_lie':
        from controllers.lie_mppi import mppi_batched
        point_lists, iters, succ, _ = mppi_batched(
            fn.TravelTimes, XP, steps=args.steps, step=STEP, temperature=TEMP,
            samples=SAMPLE_NUM, horizon=HORIZON)
        paths = [torch.cat(points).cpu().numpy() for points in point_lists]
    else:
        paths, iters, succ = MPPI_batched(XP.clone(), args.steps)
    if 'cuda' in args.device:
        torch.cuda.synchronize(args.device)
    t_plan += time.time() - ta
    tb = time.time()
    clears = POOL.map(path_clearance, paths) if POOL is not None else [path_clearance(P) for P in paths]
    for b, i in enumerate(idx):
        P = paths[b]
        clear = clears[b]
        coll = bool((clear < 0).any())
        start, goal = arr[i, :6], arr[i, 6:]
        dP = np.diff(P, axis=0)
        trans_len = float(np.linalg.norm(dP[:, :3], axis=1).sum())
        rot_chart = float(np.linalg.norm(dP[:, 3:], axis=1).sum())
        rot_geo = float(geodesic_angle(P[:-1, 3:], P[1:, 3:]).sum() / TWO_PI)
        records.append({
            'case': int(i), 'converged': bool(succ[b]), 'collision': coll,
            'success': bool(succ[b]) and not coll, 'iters': int(iters[b]) if succ[b] else None,
            'n_waypoints': int(len(P)),
            'trans_len': trans_len, 'rot_len_chart': rot_chart, 'rot_len_geo': rot_geo,
            'straight_trans': float(np.linalg.norm(goal[:3] - start[:3])),
            'straight_rot_chart': float(np.linalg.norm(goal[3:] - start[3:])),
            'straight_rot_geo': float(geodesic_angle(start[None, 3:], goal[None, 3:])[0] / TWO_PI),
            'min_clearance': float(clear.min()), 'frac_clear_lt_0p01': float((clear < 0.01).mean()),
            'theta_start_deg': float(np.linalg.norm(start[3:]) * 360), 'theta_goal_deg': float(np.linalg.norm(goal[3:]) * 360),
        })
        rec = records[-1]
        rec['_path'] = P
    t_check += time.time() - tb

# tau along each path (start -> goal) for monotonicity, one batched pass
allw, owner = [], []
for k, rec in enumerate(records):
    P = rec.pop('_path')
    g = np.repeat(P[-1:], len(P), 0)
    allw.append(np.concatenate([P, g], 1))
    owner.append(np.full(len(P), k))
W = torch.tensor(np.concatenate(allw), dtype=torch.float32) if allw else torch.zeros(0, 12)
owner = np.concatenate(owner) if owner else np.zeros(0, int)
taus = []
with torch.no_grad():
    for s in range(0, len(W), 20000):
        taus.append(fn.TravelTimes(W[s:s + 20000].to(args.device)).detach().cpu().numpy())
taus = np.concatenate(taus) if taus else np.zeros(0)
for k, rec in enumerate(records):
    tk = taus[owner == k]
    d = np.diff(tk[:-1]) if len(tk) > 2 else np.zeros(0)      # exclude the appended goal jump
    rec['tau_start'] = float(tk[0])
    rec['tau_frac_increase'] = float((d > 1e-4).mean()) if len(d) else 0.0

R = records
nok = len(R)


def rate(key):
    return float(np.mean([r[key] for r in R])) if R else None


def med_succ(key):
    v = [r[key] for r in R if r['success'] and r[key] is not None]
    return float(np.median(v)) if v else None


summary = {
    'dataPath': args.dataPath, 'kind': args.kind, 'checkpoint': args.checkpoint,
    'group_conv': bool(args.group_conv), 'planner': args.planner,
    'seed': args.seed, 'batch': args.batch, 'steps': args.steps,
    'step': STEP, 'temperature': TEMP, 'samples': SAMPLE_NUM, 'horizon': HORIZON,
    'sampling_t': STEP if args.planner == 'mppi_lie' else None,
    'distance': 'normalized product geodesic' if args.group_conv or args.planner != 'mppi' else 'chart Euclidean',
    'n_cases': n, 'n_endpoints_ok': nok, 'success_rate': rate('success'),
    'collision_rate': rate('collision'), 'nonconv_rate': float(np.mean([not r['converged'] for r in R])) if R else None,
    'median_iters_success': med_succ('iters'),
    'median_trans_ratio_success': float(np.median([r['trans_len'] / max(r['straight_trans'], 1e-3) for r in R if r['success']])) if any(r['success'] for r in R) else None,
    'median_rotgeo_ratio_success': float(np.median([r['rot_len_geo'] / max(r['straight_rot_geo'], 1e-3) for r in R if r['success']])) if any(r['success'] for r in R) else None,
    'median_rot_len_geo_success': med_succ('rot_len_geo'), 'median_trans_len_success': med_succ('trans_len'),
    'median_min_clearance_success': med_succ('min_clearance'),
    'median_tau_frac_increase': float(np.median([r['tau_frac_increase'] for r in R])) if R else None,
    'seconds': {'filter': t_filter, 'plan_gpu': t_plan, 'collision_cpu': t_check},
}
save_json({'summary': summary, 'records': R}, args.out)
if POOL is not None:
    POOL.close()
print(json.dumps(summary))
