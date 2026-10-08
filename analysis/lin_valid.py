"""Model-independent per-case labels for a test set: is the straight line from
start to goal (50-step linear interpolation in the normalized 6-D chart, as
evaluate_training_3d_batched.linear_interp_traj) collision-free, plus the
endpoint clearances and the start/goal rotation geometry.

    python analysis/lin_valid.py testing_data/3dshape/Tshape3d_env1 [...more sets]
Writes results/compare/linvalid/<set>.json.
"""
import os
import sys
import json
import multiprocessing as mp

import numpy as np
import igl
from scipy.spatial import cKDTree

sys.path.append('.')
from dataprocessing.preprocess_obj import load_obj, sample_surface_points, _rotvec_to_matrix_np  # noqa: E402
from analysis.common import geodesic_angle, TWO_PI  # noqa: E402

_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
G = {}


def setup(meta):
    def _resolve(p):
        return p if os.path.exists(p) else os.path.join('./datasets/3dshape', os.path.basename(p))
    es, ec = float(meta['env_scale']), np.asarray(meta['env_center'], dtype=np.float64)
    V, F, _ = load_obj(_resolve(meta['shape_obj']))
    G['V'] = np.ascontiguousarray((V - 0.5 * (V.min(0) + V.max(0))) / es * float(meta['shape_scale']), dtype=np.float64)
    G['F'] = np.ascontiguousarray(F, dtype=np.int64)
    G['R'] = float(np.linalg.norm(G['V'], axis=1).max())
    Ve, Fe, _ = load_obj(_resolve(meta['env_obj']))
    np.random.seed(0)
    G['env'] = np.ascontiguousarray(sample_surface_points((Ve - ec) / es, Fe, 50000), dtype=np.float64)
    G['tree'] = cKDTree(G['env'])


def min_sd(P):
    P = np.asarray(P, dtype=np.float64)
    out = np.full(len(P), 0.03)
    for k, sel in enumerate(G['tree'].query_ball_point(P[:, :3], r=G['R'] + 0.03)):
        if not sel:
            continue
        Rm = _rotvec_to_matrix_np(P[k, 3:6] * TWO_PI)
        S = igl.signed_distance(np.ascontiguousarray((G['env'][sel] - P[k, :3]) @ Rm), G['V'], G['F'], _SDF_SIGN)[0]
        out[k] = min(float(S.min()), 0.03)
    return out


def case(row):
    a, b = row[:6], row[6:]
    t = np.linspace(0, 1, 50)[:, None]
    c = min_sd((1 - t) * a + t * b)
    return {'endpoints_ok': bool(c[0] >= 0 and c[-1] >= 0), 'lin_valid': bool((c >= 0).all()),
            'lin_min_clear': float(c.min()), 'start_clear': float(c[0]), 'goal_clear': float(c[-1])}


if __name__ == '__main__':
    for d in sys.argv[1:]:
        name = os.path.basename(d.rstrip('/'))
        out = os.path.join('results/compare/linvalid', name + '.json')
        if os.path.exists(out):
            print('skip', name); continue
        meta = json.load(open(os.path.join(d, 'meta.json')))
        setup(meta)
        arr = np.load(os.path.join(d, 'sampled_points.npy')).astype(np.float64)
        with mp.get_context('fork').Pool(8) as pool:
            recs = pool.map(case, list(arr))
        geo = geodesic_angle(arr[:, 3:6], arr[:, 9:12]).numpy()
        for i, r in enumerate(recs):
            r.update({'case': i, 'trans_dist': float(np.linalg.norm(arr[i, 6:9] - arr[i, :3])),
                      'rot_geo_deg': float(np.degrees(geo[i])),
                      'rot_chart': float(np.linalg.norm(arr[i, 9:12] - arr[i, 3:6])),
                      'theta_start_deg': float(np.linalg.norm(arr[i, 3:6]) * 360),
                      'theta_goal_deg': float(np.linalg.norm(arr[i, 9:12]) * 360)})
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(recs, open(out, 'w'))
        ok = [r for r in recs if r['endpoints_ok']]
        print('{}: {} cases, endpoints ok {}, lin valid {} ({:.1%})'.format(
            name, len(recs), len(ok), sum(r['lin_valid'] for r in ok),
            np.mean([r['lin_valid'] for r in ok]) if ok else 0))
