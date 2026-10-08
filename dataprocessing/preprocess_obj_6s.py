"""Six-speed preprocessing for the ``lie_metric_6s`` pipeline.

Runs the standard ``preprocess_obj.py`` sampling (same CLI, same output files)
and then adds ``speeds_6.npy``: one speed per configuration-space dimension, for
both endpoints of every pair, from the clearance *along that dimension*:

    trans x, y, z      -- how far the shape can translate along the world axis
                          (min over +/-) before it touches an environment point.
                          The shape translated by s*u reaches env point e exactly
                          when the line e + s*u crosses a shape boundary triangle.
    rot body 1, 2, 3   -- how far the shape can rotate about its body axis R e_i
                          through its centre (min over +/-) before it touches an
                          environment point.  Each env point sweeps a circle about
                          that axis, intersected with each triangle in closed form.

Both are exact against the env point cloud (the same obstacle representation the
collision test in preprocess_obj.py uses).  The axes match the gradient the
lie_metric_6s loss weights: the world-frame position gradient dtau/dp and the
body-frame Lie rotation gradient.  The minimum over both signs is taken because
the metric M = diag(S_1^2, ..., S_6^2) is quadratic and cannot tell them apart.

Normalization is the same as speed_dists / speed_angles in preprocess_obj.py:
    trans: clip(clearance / margin, offset / margin, 1)
    rot:   clearance / pi                         (capped further in training)

    speeds_6.npy (N, 12) = [x0: sx, sy, sz, s_r1, s_r2, s_r3 | x1: same]

Usage:
    # full preprocessing (every preprocess_obj.py flag passes through) + speeds_6.npy
    python dataprocessing/preprocess_obj_6s.py --env datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Tshape3d.obj --out datasets/3dshape/Tshape3d_env1_6s \
        --num_samples 400000 --device cuda:2

    # reuse the pairs of an existing preprocess_obj.py dataset (no resampling):
    # copies its files into --out and adds speeds_6.npy there
    python dataprocessing/preprocess_obj_6s.py --from_dataset datasets/3dshape/Tshape3d_env1 \
        --out datasets/3dshape/Tshape3d_env1_6s --device cuda:2
"""

import sys
sys.path.append('.')

import os
import json
import time
import shutil
import argparse

import numpy as np
import torch

from dataprocessing import preprocess_obj as po

# Barycentric slack, so a ray / arc through an edge shared by two triangles
# still registers on one of them.
TOL = 1e-6


def resolve_mesh(path):
    """meta.json stores the generating container's absolute paths; fall back to
    the local datasets dir by basename (as the evaluate scripts do)."""
    if os.path.exists(path):
        return path
    alt = os.path.join('./datasets/3dshape', os.path.basename(path))
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError('{} (and {})'.format(path, alt))


def shape_faces_from_meta(meta):
    """(F, 3, 3) shape boundary triangles in the normalized local frame, built
    exactly as preprocess_obj.main() places the shape."""
    V, F, _ = po.load_obj(resolve_mesh(meta['shape_obj']))
    V = (V - 0.5 * (V.min(axis=0) + V.max(axis=0))) / meta['env_scale'] * meta['shape_scale']
    if meta.get('two_d'):
        z = V[:, 2]
        z_ext = float(z.max() - z.min())
        V[:, 2] = ((z - 0.5 * (z.max() + z.min())) * (po.TWOD_SHAPE_THICKNESS / z_ext)
                   if z_ext > 1e-12 else 0.0)
    return torch.tensor(V[F], dtype=torch.float32)


def _line_hits(P, d, A, E1, E2):
    """|s| at which lines P + s*d cross triangles (A, A+E1, A+E2); inf if never.
    P (B,K,1,3), d broadcastable to (B,1,1,3), A/E1/E2 (B,1,F,3) -> (B,K,F)."""
    h = torch.linalg.cross(d.expand_as(E2), E2, dim=-1)
    det = (E1 * h).sum(-1)                                      # (B,1,F)
    par = det.abs() < 1e-12                                     # line parallel to plane
    inv = 1.0 / torch.where(par, torch.ones_like(det), det)
    S = P - A                                                   # (B,K,F,3)
    u = inv * (S * h).sum(-1)
    Q = torch.linalg.cross(S, E1.expand_as(S), dim=-1)
    v = inv * (Q * d).sum(-1)
    s = inv * (Q * E2).sum(-1)
    hit = ~par & (u >= -TOL) & (v >= -TOL) & (u + v <= 1 + TOL)
    return torch.where(hit, s.abs(), torch.full_like(s, float('inf')))


def _arc_hits(P, axis, A, E1, E2):
    """Smallest |psi| in [0, pi] at which points P, rotated by psi about ``axis``
    (through the origin), land on triangle (A, A+E1, A+E2); inf if never.
    P (B,K,3), axis (B,3) unit, A/E1/E2 (B,1,F,3) -> (B,K,F)."""
    ax = axis[:, None, :]                                       # (B,1,3)
    par = (P * ax).sum(-1, keepdim=True)                        # (B,K,1)
    perp = P - par * ax
    rho = perp.norm(dim=-1, keepdim=True)                       # circle radius
    u = perp / rho.clamp_min(1e-12)
    v = torch.linalg.cross(ax.expand_as(u), u, dim=-1)
    # circle: c + rho*(cos(psi) u + sin(psi) v), c = par*axis.  On the plane
    # n.(x - A) = 0 this is  al*cos(psi) + be*sin(psi) = -k.
    c = (par * ax)[:, :, None, :]                               # (B,K,1,3)
    n = torch.linalg.cross(E1, E2, dim=-1)                      # (B,1,F,3)
    k = ((c - A) * n).sum(-1)                                   # (B,K,F)
    al = rho * (u[:, :, None, :] * n).sum(-1)
    be = rho * (v[:, :, None, :] * n).sum(-1)
    m = torch.sqrt(al**2 + be**2)
    meets = (m > 1e-12) & (k.abs() <= m)
    gam = torch.atan2(be, al)
    dl = torch.arccos((-k / m.clamp_min(1e-12)).clamp(-1.0, 1.0))
    psi = torch.stack((gam + dl, gam - dl), dim=-1)             # (B,K,F,2)

    q = (c[..., None, :] + rho[..., None, None]
         * (torch.cos(psi)[..., None] * u[:, :, None, None, :]
            + torch.sin(psi)[..., None] * v[:, :, None, None, :]))   # (B,K,F,2,3)
    w = q - A[..., None, :]
    e1, e2 = E1[..., None, :], E2[..., None, :]
    d00, d01, d11 = (e1 * e1).sum(-1), (e1 * e2).sum(-1), (e2 * e2).sum(-1)
    d20, d21 = (w * e1).sum(-1), (w * e2).sum(-1)
    den = d00 * d11 - d01**2
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    hit = meets[..., None] & (bv >= -TOL) & (bw >= -TOL) & (bv + bw <= 1 + TOL)
    ang = torch.atan2(torch.sin(psi), torch.cos(psi)).abs()     # wrapped to [0, pi]
    return torch.where(hit, ang, torch.full_like(ang, float('inf'))).amin(dim=-1)


def per_axis_clearance(configs, face_verts_local, env_points, margin, device,
                       batch_size=512):
    """Clearance along each configuration dimension.

    configs          : (N, 6) raw (x, y, z, rotvec) with the rotvec in radians
    face_verts_local : (F, 3, 3) shape boundary triangles, local frame
    env_points       : (E, 3) environment points

    Returns (N, 6): translation along world x, y, z (capped at ``margin``) |
    rotation about body axes 1, 2, 3 in radians (capped at pi).
    """
    faces = face_verts_local.to(device)
    env = env_points.to(device)
    R_shape = faces.reshape(-1, 3).norm(dim=1).max()
    eye = torch.eye(3, device=device)
    out = torch.empty(configs.shape[0], 6)
    for i in range(0, configs.shape[0], batch_size):
        cfg = configs[i:i + batch_size].to(device)
        B = cfg.shape[0]
        t = cfg[:, :3]
        R = po.rotvec_to_matrix(cfg[:, 3:6])
        tri = torch.einsum('bij,fkj->bfki', R, faces)          # (B,F,3,3) about the centre
        A = tri[:, None, :, 0]
        E1 = tri[:, None, :, 1] - A
        E2 = tri[:, None, :, 2] - A

        # Only env points within R_shape + margin of the centre can be reached by
        # a translation shorter than margin or by any rotation (which keeps |e - t|).
        dist = torch.cdist(t, env)                             # (B,E)
        reach = R_shape + margin
        K = int((dist <= reach).sum(dim=1).max().clamp(min=1))
        dk, idx = torch.topk(dist, K, dim=1, largest=False)
        P = env[idx] - t[:, None]                              # (B,K,3) about the centre
        pad = (dk > reach)[..., None]                          # (B,K,1)

        for j in range(3):
            s = _line_hits(P[:, :, None], eye[j].view(1, 1, 1, 3), A, E1, E2)
            out[i:i + B, j] = s.masked_fill(pad, float('inf')).amin(dim=(1, 2)).clamp(max=margin).cpu()
        for j in range(3):
            psi = _arc_hits(P, R[:, :, j], A, E1, E2)
            out[i:i + B, 3 + j] = psi.masked_fill(pad, float('inf')).amin(dim=(1, 2)).clamp(max=np.pi).cpu()
    return out


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--from_dataset', default=None,
                     help='Existing preprocess_obj.py dataset whose pairs to reuse '
                          '(no resampling): its files are copied into --out and '
                          'speeds_6.npy is added there.')
    own.add_argument('--speed_batch', type=int, default=512,
                     help='Configs per GPU batch in the per-axis clearance pass.')
    args, rest = own.parse_known_args()
    # The preprocess_obj.py flags this script also reads (defaults match it).
    io = argparse.ArgumentParser(add_help=False)
    io.add_argument('--out', default='datasets/3dshape/rectangle_env1')
    io.add_argument('--device', default='cuda')
    io.add_argument('--testing_data', action='store_true')
    io_args, _ = io.parse_known_args(rest)
    out = io_args.out

    if args.from_dataset:
        os.makedirs(out, exist_ok=True)
        for f in sorted(os.listdir(args.from_dataset)):
            if f.endswith('.npy') or f == 'meta.json':
                shutil.copy2(os.path.join(args.from_dataset, f), os.path.join(out, f))
        print('Copied {} -> {}'.format(args.from_dataset, out))
    else:
        if '-h' in rest or '--help' in rest:
            print(own.format_help())
        sys.argv = [sys.argv[0]] + rest
        po.main()
        if io_args.testing_data:
            print('--testing_data: start/goal pairs only, speeds_6.npy not needed.')
            return

    with open(os.path.join(out, 'meta.json')) as f:
        meta = json.load(f)
    margin, offset = float(meta['margin']), float(meta['offset'])
    pairs = np.load(os.path.join(out, 'sampled_points.npy'))       # (N, 12), rotvec / 2pi
    configs = pairs.reshape(-1, 6).astype(np.float32)             # rows x0_0, x1_0, x0_1, ...
    configs[:, 3:6] *= 2 * np.pi
    env_points = torch.tensor(np.load(os.path.join(out, 'env.npy')), dtype=torch.float32)
    faces = shape_faces_from_meta(meta)

    print('Per-axis clearance for {} configs ({} pairs)  margin={}  offset={} ...'.format(
        configs.shape[0], pairs.shape[0], margin, offset))
    t0 = time.time()
    clear = per_axis_clearance(torch.from_numpy(configs), faces, env_points, margin,
                               io_args.device, args.speed_batch).numpy()
    print('done in {:.1f}s'.format(time.time() - t0))

    speeds = np.concatenate((np.clip(clear[:, :3] / margin, offset / margin, 1.0),
                             clear[:, 3:] / np.pi), axis=1)
    speeds = speeds.reshape(-1, 12).astype(np.float32)            # [x0 (6) | x1 (6)]
    np.save(os.path.join(out, 'speeds_6'), speeds)

    meta['speeds_6'] = {
        'layout': '[x0: sx, sy, sz, s_r1, s_r2, s_r3 | x1: same]',
        'trans_axes': 'world x, y, z; clip(clearance / margin, offset / margin, 1)',
        'rot_axes': 'body axes R e_i through the shape centre; clearance (rad) / pi',
    }
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    names = ['sx', 'sy', 'sz', 's_r1', 's_r2', 's_r3']
    for e, tag in ((0, 'x0'), (1, 'x1')):
        s = speeds[:, 6 * e:6 * e + 6]
        print('  {}  '.format(tag) + '  '.join(
            '{}: mean {:.3f} sat {:.2f}'.format(n, s[:, k].mean(), (s[:, k] >= 1).mean())
            for k, n in enumerate(names)))
    print('Wrote speeds_6.npy {} to {}'.format(speeds.shape, out))


if __name__ == '__main__':
    main()
