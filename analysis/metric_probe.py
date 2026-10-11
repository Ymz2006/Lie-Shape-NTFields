"""Learned local metric in free space: slope of tau(g, g exp(delta e_i)) in delta, per body
twist axis e_i (translation x,y,z per env unit; rotation about body x,y,z per turn).

    python analysis/metric_probe.py --kind twist --checkpoint <pt> --out <json>

Targets in free space (S_dist = S_angle = 1): lie_metric |e_i| = 1 for every axis;
lie_twist sqrt(G0_ii) of the checkpoint's twist metric.
"""
import sys, math, json, argparse
sys.path.append('.')
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from analysis.common import load_model

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True, choices=['metric', 'shape', 'lie', 'twist', 'lie_rnl'])
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--n', type=int, default=400)
args = ap.parse_args()
TWO_PI = 2 * math.pi
torch.manual_seed(0); np.random.seed(0)

model, fn = load_model(args.kind, args.checkpoint, args.device)
G0 = np.asarray(getattr(model, 'G0', np.eye(6)))

d = 'datasets/3dshape/Tshape3d_env1'   # both endpoints of the training pairs
pts = np.load(d + '/sampled_points.npy').reshape(-1, 6)
sd = np.load(d + '/speed_dists.npy').reshape(-1)
sa = np.load(d + '/speed_angles.npy').reshape(-1)
free = (sd >= 1.0) & (sa >= 1.0 / 6.0)                     # both training speeds saturate at 1
q = pts[free][:args.n]
rv = q[:, 3:] * TWO_PI
ok = np.linalg.norm(rv, axis=1) < 0.8 * math.pi             # stay away from the chart's cut locus
q = q[ok]
Rq = Rot.from_rotvec(q[:, 3:] * TWO_PI)


def move(i, delta):
    e = np.zeros(6); e[i] = delta
    p = q[:, :3] + Rq.apply(np.tile(e[:3], (len(q), 1)))
    R2 = Rq * Rot.from_rotvec(TWO_PI * np.tile(e[3:], (len(q), 1)))
    return np.concatenate([p, R2.as_rotvec() / TWO_PI], 1)


def tau(a, b):
    X = torch.tensor(np.concatenate([a, b], 1), dtype=torch.float32, device=args.device)
    with torch.no_grad():
        t1 = fn.TravelTimes(X).cpu().numpy()
        X2 = torch.cat([X[:, 6:], X[:, :6]], 1)
        t2 = fn.TravelTimes(X2).cpu().numpy()
    return 0.5 * (t1 + t2)


d1, d2 = 0.01, 0.02
names = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
res = {'n_configs': int(len(q)), 'G0_diag_sqrt': np.sqrt(np.diag(G0)).tolist(), 'axes': {}}
for i, nm in enumerate(names):
    s = (tau(q, move(i, d2)) - tau(q, move(i, d1))) / (d2 - d1)
    res['axes'][nm] = {'median': float(np.median(s)), 'q25': float(np.percentile(s, 25)),
                       'q75': float(np.percentile(s, 75))}
med = np.array([res['axes'][k]['median'] for k in names])
res['rot_over_trans'] = (med[3:] / med[:3].mean()).tolist()        # learned rotation cost per turn / translation cost per unit
res['rot_anisotropy_wz_over_wy'] = float(med[5] / med[4])
json.dump(res, open(args.out, 'w'), indent=1)
print(args.checkpoint)
print('  n={}  slopes (median): '.format(len(q)) + ' '.join('{}={:.3f}'.format(k, m) for k, m in zip(names, med)))
print('  rot/trans per axis: ' + ' '.join('{:.3f}'.format(x) for x in res['rot_over_trans'])
      + '   target sqrt(G0_ii): ' + ' '.join('{:.3f}'.format(x) for x in np.sqrt(np.diag(G0))[3:]))
print('  wz/wy = {:.3f}  (target {:.3f})'.format(res['rot_anisotropy_wz_over_wy'], math.sqrt(G0[5, 5] / G0[4, 4])))
