"""Build the held-out data for the metric / shape / lie comparison (Tshape3d_env1).

    python analysis/make_sets.py --out results/compare/sets --device cuda:0

Writes (all configs normalized: x, y, z, rotvec / 2 pi):
  probe_set.npz    held-out configurations labelled with clearance (dist, angle) and
                   the clearance-reduction directions (trans_n, rot_n): half drawn
                   from the training sampler, half near the obstacle surfaces so the
                   tight-clearance bins are populated.  Used for the binned speed
                   analysis.  Plus a pool of generic goals.
  free_pairs.npz   pairs whose endpoints AND interior (coordinate line and SO(3)
                   geodesic) have saturated clearance (S* = 1): free-space metric
                   identification.
  rot_pairs.npz    pure-rotation pairs at positions where every orientation is free:
                   'generic' random pairs, 'wrap' pairs that are close on SO(3) but
                   whose rotation vectors lie on opposite sides of the cut locus, and
                   'control' pairs with the same geodesic angle on one axis without
                   crossing it.
  cut_set.npz      configurations with rotation angle >= 150 deg and their
                   equivalent representation across the cut locus (r - r/|r|).
And three planning test sets under testing_data/3dshape/ (same layout as the
preprocess testing data; endpoints free with clearance > --test_offset):
  Tshape3d_env1_rotheavy   |dp| <= 0.05, geodesic rotation >= 90 deg
  Tshape3d_env1_bigangle   both endpoints with rotation angle in [135, 180) deg
  Tshape3d_env1_transonly  identical rotation, |dp| >= 0.3 (control)
"""
import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch

sys.path.append('.')
from analysis.common import (Scene, TWO_PI, MARGIN, random_rotvec, so3_exp, so3_log,  # noqa: E402
                             geodesic_angle, save_json)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='results/compare/sets')
ap.add_argument('--device', default='cpu')
ap.add_argument('--n_probe', type=int, default=40000, help='candidates per probe half')
ap.add_argument('--n_test', type=int, default=200)
ap.add_argument('--test_offset', type=float, default=0.005)
ap.add_argument('--chunk', type=int, default=200)
ap.add_argument('--only', default='', help='comma list of parts to build')
ap.add_argument('--probe_tag', default='', help='write probe_set<tag>.npz (extra probe rounds)')
ap.add_argument('--no_near', action='store_true', help='probe: uniform candidates only')
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--cell', default='Tshape3d_env1', help='<shape>_<env>: scene for the probe / free / rot / cut parts')
ap.add_argument('--free_target', type=int, default=1500)
ap.add_argument('--free_tries', type=int, default=12)
args = ap.parse_args()
torch.set_num_threads(8)

os.makedirs(args.out, exist_ok=True)
gen = torch.Generator().manual_seed(args.seed)
t0 = time.time()
_shape, _env = args.cell.split('_')
scene = Scene(env_obj='datasets/3dshape/{}.obj'.format(_env), shape_obj='datasets/3dshape/{}.obj'.format(_shape),
              device=args.device)
print('scene ready {:.1f}s  R_shape={:.4f}'.format(time.time() - t0, scene.shape_radius))
parts = set(args.only.split(',')) if args.only else {'probe', 'free', 'rot', 'cut', 'plan'}


def clear(q):
    return scene.clearance(q, chunk=args.chunk)


def rotvec_compose(r_left_rad, r):
    """Exp(r_left) Exp(2 pi r) -> normalized rotvec."""
    R = so3_exp(r_left_rad.double()) @ so3_exp(TWO_PI * r.double())
    return (so3_log(R) / TWO_PI).float()


# ----------------------------------------------------------------- probe set
if 'probe' in parts:
    n = args.n_probe
    qa = scene.sample_configs(n, gen)
    env = scene.env_points
    idx = torch.randint(0, len(env), (n,), generator=gen)
    off = torch.randn(n, 3, generator=gen)
    off = off / off.norm(dim=1, keepdim=True) * torch.rand(n, 1, generator=gen) * 0.10
    qb = torch.empty(n, 6)
    qb[:, :3] = env[idx] + off
    he = torch.tensor(scene.half_extent, dtype=torch.float32)
    qb[:, :3] = torch.max(torch.min(qb[:, :3], he), -he)
    qb[:, 3:] = random_rotvec(n, gen) / TWO_PI
    q = torch.cat([qa, qb]) if not args.no_near else qa
    src = torch.cat([torch.zeros(n), torch.ones(n)]) if not args.no_near else torch.zeros(n)
    t1 = time.time()
    c = clear(q)
    keep = c['free'] & (c['dist'] > 0.001)
    print('probe: {} / {} kept ({:.0f}s)'.format(int(keep.sum()), len(q), time.time() - t1))
    np.savez(os.path.join(args.out, 'probe_set{}.npz'.format(args.probe_tag)), q=q[keep].numpy(),
             dist=c['dist'][keep].numpy(), angle=c['angle'][keep].numpy(),
             trans_n=c['trans_n'][keep].numpy(), rot_n=c['rot_n'][keep].numpy(),
             near=src[keep].numpy())
    # generic goal pool: moderate clearance, spread over the box
    if args.probe_tag:
        sys.exit(0)
    g = scene.sample_configs(4000, gen)
    cg = clear(g)
    ok = cg['free'] & (cg['dist'] > 0.02) & (cg['angle'] > math.radians(10))
    goals = g[ok][:16]
    np.save(os.path.join(args.out, 'goal_pool.npy'), goals.numpy())
    print('goal pool:', goals.shape)

# ----------------------------------------------------------------- free pairs
if 'free' in parts:
    pairs = []
    tries = 0
    while sum(len(p) for p in pairs) < args.free_target and tries < args.free_tries:
        tries += 1
        q0 = scene.sample_configs(6000, gen)
        dp = torch.randn(6000, 3, generator=gen)
        dp = dp / dp.norm(dim=1, keepdim=True) * torch.rand(6000, 1, generator=gen) * 0.15
        dr = random_rotvec(6000, gen, 0.0, math.radians(60))
        q1 = torch.empty_like(q0)
        q1[:, :3] = q0[:, :3] + dp
        q1[:, 3:] = rotvec_compose(dr, q0[:, 3:])
        # interior checks: coordinate line at t=1/3, 2/3 and the SO(3) geodesic midpoint
        mids = []
        for t in (1 / 3, 2 / 3):
            m = q0 + t * (q1 - q0)
            mids.append(m)
        mg = q0.clone()
        mg[:, :3] = 0.5 * (q0[:, :3] + q1[:, :3])
        mg[:, 3:] = rotvec_compose(0.5 * dr, q0[:, 3:])
        mids.append(mg)
        allq = torch.cat([q0, q1] + mids)
        he = torch.tensor(scene.half_extent, dtype=torch.float32)
        inbox = (allq[:, :3].abs() <= he).all(1).view(5, -1).all(0)
        c = clear(allq)
        sat = (c['free'] & (c['dist'] >= MARGIN) & (c['angle'] >= math.radians(30))).view(5, -1).all(0)
        ok = sat & inbox
        pairs.append(torch.cat([q0[ok], q1[ok]], 1))
        print('free pairs try {}: +{}'.format(tries, int(ok.sum())))
    P = torch.cat(pairs)[:args.free_target]
    np.save(os.path.join(args.out, 'free_pairs.npy'), P.numpy())
    print('free pairs:', P.shape)

# ----------------------------------------------------------------- rotation-only pairs
if 'rot' in parts:
    # Pure rotations at a fixed position.  No position in env1 lets the T-shape turn
    # freely (R = 0.2 in env units), so pairs are kept when both endpoints are free
    # with clearance > 0.005, and the comparisons are made pairwise:
    #   generic : random start / goal rotations
    #   wrap    : r0 = th0 u, r1 = -th1 u, th0, th1 in [130, 175] deg -- close on SO(3)
    #             (geodesic 2 pi - th0 - th1) but far apart in the chart
    #   control : the same position, start and axis, goal th0 - gap on the near side,
    #             same geodesic angle, no cut-locus crossing
    def make(n):
        x = scene.sample_configs(n, gen)
        u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=1, keepdim=True)
        th0 = torch.deg2rad(130 + 45 * torch.rand(n, 1, generator=gen))
        th1 = torch.deg2rad(130 + 45 * torch.rand(n, 1, generator=gen))
        gap = TWO_PI - th0 - th1
        rg0 = random_rotvec(n, gen) / TWO_PI
        rg1 = random_rotvec(n, gen) / TWO_PI
        rw0, rw1 = th0 * u / TWO_PI, -th1 * u / TWO_PI
        rc1 = (th0 - gap) * u / TWO_PI
        return x[:, :3], rg0, rg1, rw0, rw1, rc1

    keep_all = {k: [] for k in ('p', 'rg0', 'rg1', 'rw0', 'rw1', 'rc1')}
    got, tries = 0, 0
    while got < 1000 and tries < 15:
        tries += 1
        p, rg0, rg1, rw0, rw1, rc1 = make(4000)
        cfgs = [torch.cat([p, r], 1) for r in (rg0, rg1, rw0, rw1, rc1)]
        cc = [clear(c) for c in cfgs]
        ok = torch.ones(len(p), dtype=torch.bool)
        for c in cc:
            ok &= c['free'] & (c['dist'] > 0.005)
        for k, v in zip(('p', 'rg0', 'rg1', 'rw0', 'rw1', 'rc1'), (p, rg0, rg1, rw0, rw1, rc1)):
            keep_all[k].append(v[ok])
        got += int(ok.sum())
        print('rot pairs try {}: {}'.format(tries, got), flush=True)
    K = {k: torch.cat(v)[:1000] for k, v in keep_all.items()}
    n = len(K['p'])
    Q0 = torch.cat([torch.cat([K['p'], K['rg0']], 1), torch.cat([K['p'], K['rw0']], 1), torch.cat([K['p'], K['rw0']], 1)])
    Q1 = torch.cat([torch.cat([K['p'], K['rg1']], 1), torch.cat([K['p'], K['rw1']], 1), torch.cat([K['p'], K['rc1']], 1)])
    kind = np.array(['generic'] * n + ['wrap'] * n + ['control'] * n)
    geo = geodesic_angle(Q0[:, 3:], Q1[:, 3:]).numpy()
    np.savez(os.path.join(args.out, 'rot_pairs.npz'), q0=Q0.numpy(), q1=Q1.numpy(), kind=kind, geo=geo)
    print('rot pairs:', n, 'per kind; wrap vs control geodesic max diff (deg):',
          float(np.degrees(np.abs(geo[n:2 * n] - geo[2 * n:])).max()), flush=True)

# ----------------------------------------------------------------- cut-locus set
if 'cut' in parts:
    n = 6000
    q = scene.sample_configs(n, gen)
    q[:, 3:] = random_rotvec(n, gen, math.radians(150), math.radians(179.5)) / TWO_PI
    c = clear(q)
    ok = c['free'] & (c['dist'] > 0.001)
    q = q[ok][:3000]
    r = q[:, 3:]
    r_alt = r - r / r.norm(dim=1, keepdim=True)          # same rotation, |r'| = 1 - |r|
    q_alt = torch.cat([q[:, :3], r_alt], 1)
    Ra, Rb = so3_exp(TWO_PI * r.double()), so3_exp(TWO_PI * r_alt.double())
    print('cut set: {} configs, max |R - R_alt| = {:.2e}'.format(len(q), (Ra - Rb).abs().max().item()))
    np.savez(os.path.join(args.out, 'cut_set.npz'), q=q.numpy(), q_alt=q_alt.numpy(),
             dist=c['dist'][ok][:3000].numpy(), angle=c['angle'][ok][:3000].numpy())


# ----------------------------------------------------------------- planning test sets
def write_test_set(name, x0, x1, c0, c1):
    d = os.path.join('testing_data/3dshape', name)
    os.makedirs(d, exist_ok=True)
    pts = torch.cat([x0, x1], 1).numpy()
    sd = np.stack([np.clip(c0['dist'].numpy() / MARGIN, 0.001 / MARGIN, 1),
                   np.clip(c1['dist'].numpy() / MARGIN, 0.001 / MARGIN, 1)], 1)
    sa = np.stack([c0['angle'].numpy() / np.pi, c1['angle'].numpy() / np.pi], 1)
    np.save(os.path.join(d, 'sampled_points.npy'), pts)
    np.save(os.path.join(d, 'speed_dists.npy'), sd)
    np.save(os.path.join(d, 'speed_angles.npy'), sa)
    np.save(os.path.join(d, 'speed.npy'), (sd + sa) / 2)
    np.save(os.path.join(d, 'normal.npy'), np.zeros((len(pts), 12), np.float32))
    np.save(os.path.join(d, 'env.npy'), scene.env_points.numpy())
    meta = json.load(open('testing_data/3dshape/Tshape3d_env1/meta.json'))
    meta.update({'offset': args.test_offset, 'testing_data': True,
                 'generator': 'analysis/make_sets.py', 'set': name})
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=2)
    print('wrote {} ({} pairs)'.format(d, len(pts)))


def build_pairs(name, make, need):
    xs0, xs1, cs0, cs1 = [], [], [], []
    got, tries = 0, 0
    while got < need and tries < 40:
        tries += 1
        x0, x1 = make(3000)
        he = torch.tensor(scene.half_extent, dtype=torch.float32)
        inbox = (x1[:, :3].abs() <= he).all(1)
        x0, x1 = x0[inbox], x1[inbox]
        c0, c1 = clear(x0), clear(x1)
        ok = c0['free'] & c1['free'] & (c0['dist'] > args.test_offset) & (c1['dist'] > args.test_offset)
        xs0.append(x0[ok]); xs1.append(x1[ok])
        cs0.append({k: v[ok] for k, v in c0.items()}); cs1.append({k: v[ok] for k, v in c1.items()})
        got += int(ok.sum())
        print('{} try {}: {} / {}'.format(name, tries, got, need))
    cat = lambda L: {k: torch.cat([d[k] for d in L])[:need] for k in L[0]}
    write_test_set(name, torch.cat(xs0)[:need], torch.cat(xs1)[:need], cat(cs0), cat(cs1))


if 'plan' in parts:
    N = args.n_test

    def make_rotheavy(n):
        x0 = scene.sample_configs(n, gen)
        dp = torch.randn(n, 3, generator=gen)
        dp = dp / dp.norm(dim=1, keepdim=True) * torch.rand(n, 1, generator=gen) * 0.05
        x1 = x0.clone(); x1[:, :3] += dp
        x1[:, 3:] = random_rotvec(n, gen) / TWO_PI
        keep = geodesic_angle(x0[:, 3:], x1[:, 3:]) >= math.radians(90)
        return x0[keep], x1[keep]

    def make_bigangle(n):
        x0 = scene.sample_configs(n, gen); x1 = scene.sample_configs(n, gen)
        x0[:, 3:] = random_rotvec(n, gen, math.radians(135), math.pi) / TWO_PI
        x1[:, 3:] = random_rotvec(n, gen, math.radians(135), math.pi) / TWO_PI
        return x0, x1

    def make_transonly(n):
        x0 = scene.sample_configs(n, gen); x1 = scene.sample_configs(n, gen)
        x1[:, 3:] = x0[:, 3:]
        keep = (x1[:, :3] - x0[:, :3]).norm(dim=1) >= 0.3
        return x0[keep], x1[keep]

    build_pairs('Tshape3d_env1_rotheavy', make_rotheavy, N)
    build_pairs('Tshape3d_env1_bigangle', make_bigangle, N)
    build_pairs('Tshape3d_env1_transonly', make_transonly, N)

print('done in {:.0f}s'.format(time.time() - t0))
