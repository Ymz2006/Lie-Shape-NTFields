"""Shared helpers for the metric / shape / lie NTField comparison (analysis/*.py).

Geometry is rebuilt exactly as dataprocessing/preprocess_obj.py builds it, so
clearance labels computed here (translational distance, rotational min-angle)
are the same quantities the training data was labelled with.  Speed targets use
the same warps as the training loops (models/*/model_train_metric.py).

Configurations are 6-D (x, y, z, r) with r = rotvec / (2 pi), as everywhere else.
"""
import os
import sys
import math
import json
import importlib

import numpy as np
import torch

sys.path.append('.')
from dataprocessing import preprocess_obj as po          # noqa: E402
from models import rnl                                    # noqa: E402

TWO_PI = 2.0 * math.pi
DIM = 6
MARGIN = 0.05            # training margin (meta.json); S*_dist saturates at d = MARGIN
TRAIN_OFFSET = 0.001     # training offset (lower clip of the distance ratio)
ANGLE_MAX = 1.0 / 6.0    # training loop: speed_angle (angle / pi) saturates at 30 deg

MODEL_PKGS = {'metric': 'metric_baseline', 'shape': 'shape_metric', 'lie': 'lie_metric'}


# --------------------------------------------------------------------------- geometry
class Scene:
    """Environment + shape in the normalized frame, as preprocess_obj.py builds them."""

    def __init__(self, env_obj='datasets/3dshape/env1.obj',
                 shape_obj='datasets/3dshape/Tshape3d.obj', device='cpu',
                 num_env_points=po.DEFAULT_ENV_POINTS, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = device
        V_env, F_env, names_env = po.load_obj(env_obj)
        bb_min, bb_max = V_env.min(axis=0), V_env.max(axis=0)
        self.env_center = 0.5 * (bb_min + bb_max)
        self.env_scale = float((bb_max - bb_min).max())
        V_env_n = (V_env - self.env_center) / self.env_scale
        sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
        self.env_points = torch.tensor(
            po.sample_surface_points(V_env_n, F_env[sample_mask], num_env_points).astype(np.float32))
        ranges = (bb_max - bb_min) / self.env_scale
        self.half_extent = ranges * 0.5 - 0.01

        V_sh, F_sh, _ = po.load_obj(shape_obj)
        self.shape_V_local = (V_sh - 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))) / self.env_scale
        self.shape_F = F_sh
        TV, TT, TF = po.tetrahedralize_shape(self.shape_V_local, F_sh, switches=po.DEFAULT_TET_SWITCHES)
        self.tets = torch.tensor(TV[TT], dtype=torch.float32)
        self.faces = torch.tensor(TV[TF], dtype=torch.float32)
        self.rad_points, self.rad_bins = po.generate_radius_surface_points(self.shape_V_local, F_sh, 1000, 10)
        self.shape_radius = float(np.linalg.norm(self.shape_V_local, axis=1).max())

        # Dense collision cloud over the FULL env mesh (obstacles + walls), as in
        # evaluate_training_3d_batched.py.
        self.collision_pts = np.ascontiguousarray(
            po.sample_surface_points(V_env_n, F_env, 50000), dtype=np.float64)
        self.shape_V64 = np.ascontiguousarray(self.shape_V_local, dtype=np.float64)
        self.shape_F64 = np.ascontiguousarray(F_sh, dtype=np.int64)

    def clearance(self, q, chunk=200):
        """q: (N, 6) normalized configs (tensor or array).  Returns dict of CPU
        tensors: free (bool), dist, angle (rad), trans_n (3), rot_n (3)."""
        q = torch.as_tensor(q, dtype=torch.float32)
        out = {k: [] for k in ('free', 'dist', 'angle', 'trans_n', 'rot_n')}
        for s in range(0, len(q), chunk):
            cfg = q[s:s + chunk].clone()
            cfg[:, 3:] *= TWO_PI
            f, d, a, _n, tn, rn = po.evaluate_placements(
                cfg, self.tets, self.faces, self.env_points, self.rad_points,
                self.rad_bins, self.device)
            out['free'].append(f); out['dist'].append(d); out['angle'].append(a)
            out['trans_n'].append(tn[:, :3]); out['rot_n'].append(rn[:, 3:])
        return {k: torch.cat(v) for k, v in out.items()}

    def sample_configs(self, n, generator=None):
        """Training-distribution sampler: position uniform in the env box, rotation
        axis uniform, angle uniform in [0, pi] (preprocess_obj._sample_configs)."""
        hx, hy, hz = (float(h) for h in self.half_extent)
        c = torch.empty(n, 6)
        c[:, 0] = (torch.rand(n, generator=generator) * 2 - 1) * hx
        c[:, 1] = (torch.rand(n, generator=generator) * 2 - 1) * hy
        c[:, 2] = (torch.rand(n, generator=generator) * 2 - 1) * hz
        c[:, 3:] = random_rotvec(n, generator) / TWO_PI
        return c


def random_rotvec(n, generator=None, angle_lo=0.0, angle_hi=math.pi):
    """Rotation vectors (radians): uniform axis, angle uniform in [angle_lo, angle_hi]."""
    ax = torch.randn(n, 3, generator=generator)
    ax = ax / ax.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ang = angle_lo + torch.rand(n, 1, generator=generator) * (angle_hi - angle_lo)
    return ax * ang


# --------------------------------------------------------------------------- targets
def warp(s):
    return s ** 2 * (2 - s) ** 2


def target_speeds(dist, angle):
    """Warped speed targets exactly as the training loops build them.
    dist: translational clearance; angle: rotational clearance (radians)."""
    dist = torch.as_tensor(dist, dtype=torch.float64)
    angle = torch.as_tensor(angle, dtype=torch.float64)
    s_d = torch.clamp(dist / MARGIN, min=TRAIN_OFFSET / MARGIN, max=1.0)
    s_a = torch.clamp((angle / math.pi) / ANGLE_MAX, min=1e-4 / ANGLE_MAX, max=1.0)
    S_dist = torch.clamp(warp(s_d), min=1e-3)
    S_angle = torch.clamp(warp(s_a), min=1e-3)
    # metric_baseline: combined speed.npy = (s_d + angle/pi) / 2, warped
    S_comb = warp((s_d + angle / math.pi) / 2)
    return S_dist, S_angle, S_comb


# --------------------------------------------------------------------------- SO(3)
def so3_exp(w):
    return rnl.so3_exp(w)


def so3_log(R):
    return rnl.so3_log(R)


def geodesic_angle(r0, r1):
    """Geodesic angle (radians) between rotations given as normalized rotvecs."""
    R0 = so3_exp(TWO_PI * torch.as_tensor(r0, dtype=torch.float64))
    R1 = so3_exp(TWO_PI * torch.as_tensor(r1, dtype=torch.float64))
    return so3_log(R0.transpose(-1, -2) @ R1).norm(dim=-1)


def jr_inv(phi):
    """Inverse right Jacobian of SO(3) at rotvec phi (B,3) radians (float64)."""
    phi = phi.double()
    theta = phi.norm(dim=-1)
    small = theta < 1e-2
    th = torch.where(small, torch.ones_like(theta), theta).clamp(max=math.pi - 1e-3)
    c = 1.0 / th ** 2 - (1.0 + torch.cos(th)) / (2.0 * th * torch.sin(th))
    c = torch.where(small, 1.0 / 12.0 + theta ** 2 / 720.0, c)
    K = rnl.hat(phi)
    I = torch.eye(3, dtype=phi.dtype).expand_as(K)
    return I + 0.5 * K + c[:, None, None] * (K @ K)


def body_rot_grad(r, g_r):
    """Body-frame (Lie) rotational gradient, same map as lie_metric.lie_rot_grad:
    grad_w = Jr^{-1}(2 pi r)^T dtau/dr.  r, g_r: (B,3)."""
    J = jr_inv(TWO_PI * torch.as_tensor(r).double())
    return torch.einsum('bij,bi->bj', J, torch.as_tensor(g_r).double())


# --------------------------------------------------------------------------- models
def load_model(kind, checkpoint, device):
    """kind in {'metric','shape','lie'} -> (Model, Function)."""
    pkg = MODEL_PKGS[kind]
    md = importlib.import_module('models.{}.model_train_metric'.format(pkg))
    m = md.Model('/tmp/analysis_model', 'datasets/3dshape/Tshape3d_env1/', DIM, [0.0] * DIM, device=device)
    m.load(checkpoint)
    m.network.eval()
    return m, m.function


def tau_and_grad(fn, X, chunk=4096, need_grad=True):
    """X: (N, 12) [x0 | x1] on any device.  Returns (tau (N,), dtau (N,12)) on CPU,
    float64.  dtau is the gradient w.r.t. both endpoints."""
    taus, grads = [], []
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk].to(fn.device if hasattr(fn, 'device') else X.device).float()
        tau, _, coords = fn.network.out(xb)
        if need_grad:
            g = torch.autograd.grad(tau.sum(), coords, create_graph=False)[0]
            grads.append(g.detach().double().cpu())
        taus.append(tau[:, 0].detach().double().cpu())
        del tau, coords
    tau = torch.cat(taus)
    return tau, (torch.cat(grads) if need_grad else None)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
