"""Relative Normal Loss (RNL) -- Shape-NTFields eqs. (6)-(7), with either the
original coordinate probes or the Lie-group probe curves of
``RNL_Lie_Group_Probes.pdf``.

Shared by ``models/shape_metric`` and ``models/lie_metric``: the probe curves
only depend on the data, and each package supplies its own speed function S_c
(the one its Eikonal term constrains).

Frames (see ``dataprocessing/preprocess_obj.py::evaluate_placements``):
    * a placement is g = (R, p), R = Exp(phi), world point = R x_local + p; the
      network sees q = (p, r) with r = phi / (2 pi).
    * ``trans_n`` = -n_ws: world-frame unit translation that closes the gap to
      the nearest obstacle fastest.
    * ``rot_n``  = unit(u_s x u_e): world-frame unit rotation axis turning the
      shape-radial direction u_s onto the obstacle-radial direction u_e, both
      measured from the placement centre p.  The rotation centre is therefore
      the body origin (c = 0 in the notes).

Probes at the band point q_s (always x0, goal x1 held fixed):
    'orig' : q_s + eps (n_trans, 0)  and  q_s + eps (0, n_rot)   (coordinate lines)
    'lie'  : g exp(eps V_t^)         and  g exp(2 pi eps V_r^)   (group curves)
             with the world normals pulled back to body twists in the tangent
             space at g:  V_t = (0, R^T n_trans),  V_r = (R^T n_rot, 0).
"""
import math
import torch

TWO_PI = 2.0 * math.pi


def hat(w):
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix [w]x."""
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    return torch.stack([torch.stack([O, -wz, wy], -1),
                        torch.stack([wz, O, -wx], -1),
                        torch.stack([-wy, wx, O], -1)], -2)


def so3_exp(w):
    """Rodrigues formula, stable for small angles. w: (..., 3) rotation vector."""
    th2 = (w * w).sum(-1, keepdim=True)[..., None]
    th = th2.clamp_min(1e-24).sqrt()
    small = th2 < 1e-8
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-24))
    K = hat(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return I + A * K + B * (K @ K)


def so3_log(R):
    """Rotation vector with angle in [0, pi]; robust at 0 and pi (via quaternion)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22
    cands = torch.stack([
        torch.stack([1 + t, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, 1 + 2 * m00 - t, m01 + m10, m02 + m20], -1),
        torch.stack([m02 - m20, m01 + m10, 1 + 2 * m11 - t, m12 + m21], -1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 + 2 * m22 - t], -1),
    ], -2)
    diag = torch.stack([t, m00, m11, m22], -1)
    idx = diag.argmax(-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(*idx.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[..., :1] < 0, -q, q)                    # w >= 0 => angle <= pi
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(s, q[..., :1])
    scale = torch.where(s < 1e-12, 2 / q[..., :1].clamp_min(1e-12), ang / s.clamp_min(1e-12))
    return scale * v


@torch.no_grad()
def make_probes(q_s, n_trans, n_rot, eps, mode, cut_thresh=math.pi):
    """Probe inputs for a batch of band points.

    q_s     : (B, 6) network inputs (p, r), r = Log(R) / (2 pi)
    n_trans : (B, 3) unit world-frame translation normal
    n_rot   : (B, 3) unit world-frame rotation axis
    eps     : probe size in normalized units (rotation probe turns 2 pi eps rad)
    mode    : 'orig' (coordinate lines) or 'lie' (group curves)
    Returns q_trans, q_rot (B, 6) and valid_rot (B,) bool (cut-locus mask; all
    True for 'orig', which never re-charts the rotation vector).
    """
    p, r = q_s[:, :3], q_s[:, 3:]
    if mode == 'orig':
        q_t = torch.cat([p + eps * n_trans, r], -1)
        q_r = torch.cat([p, r + eps * n_rot], -1)
        return q_t, q_r, torch.ones_like(p[:, 0], dtype=torch.bool)
    assert mode == 'lie', mode

    # Double precision for the Exp/Log round trip (probe offsets are ~1e-2).
    p64, w_s = p.double(), TWO_PI * r.double()
    R = so3_exp(w_s)
    Rt = R.transpose(-1, -2)
    # World normals -> body twists in the tangent space at g (eq. 1, c = 0).
    v_t = (Rt @ n_trans.double()[..., None])[..., 0]       # V_t = (0, R^T n_trans)
    w_b = (Rt @ n_rot.double()[..., None])[..., 0]         # V_r = (R^T n_rot, 0)

    # a(eps) = g exp(eps V_t^) = (R, p + eps R v_t)  ==  (R, p + eps n_trans)
    q_t = torch.cat([p64 + eps * (R @ v_t[..., None])[..., 0], w_s / TWO_PI], -1)

    # b(2 pi eps) = g exp(2 pi eps V_r^) = (R Exp(2 pi eps w_b), p)
    #             == (Exp(2 pi eps n_rot) R, p): rotation about the world axis
    #             n_rot through the body origin, which stays fixed.
    R_new = R @ so3_exp(TWO_PI * eps * w_b)
    w_new = so3_log(R_new)
    q_r = torch.cat([p64, w_new / TWO_PI], -1)

    # Cut locus: a legal step moves the rotvec by <= pi^2 eps; wrapping past
    # angle pi flips it by ~2 pi.  Mask those probes (eq. 8).
    valid = (w_new - w_s).norm(dim=-1) < cut_thresh
    return q_t.to(q_s.dtype), q_r.to(q_s.dtype), valid


def rnl_terms(S0, S1, S_star, eps, rho, G_max, valid=None):
    """Per-sample eq. (7) for one block:
        G = (S(q_probe) - S(q_s)) / eps
        L = 1[S* < 1] * valid * (rho - clip(G, rho, G_max))^2
    Returns (L, G, active)."""
    G = (S1 - S0) / eps
    L = (rho - torch.clamp(G, min=rho, max=G_max)) ** 2
    active = (S_star < 1).to(L.dtype)
    if valid is not None:
        active = active * valid.to(L.dtype)
    return active * L, G, active


# Values from the original Shape-NTFields code
# (shape-ntfields/models/metric/model_function_metric.py; names there in brackets).
DEFAULTS = {'mode': 'none',          # 'none' | 'orig' | 'lie'
            'eps': 0.02,             # [step] probe size, normalized units
                                     #   (the Lie rotation probe turns 2 pi eps rad)
            'rho_trans': -1 / 0.05,  # [rate_trans] target slope: speed drops at
            'rho_rot': -1 / (1 / 6), # [rate_rot]   least this fast along the probe
            'G_max': 10.0,           # [dsdn_clip] slope where the penalty saturates
            # [err_weight] trans/rot weights each ramp linearly
            #   weight * clip((epoch - ramp_start) / ramp_len, 0, 1)
            'weight': 1e-4,
            'ramp_start': 1500,
            'ramp_len': 2000,
            # [tr_cap_scale] causality-weighted RNL contribution is capped at
            # cap_ratio x the Eikonal one (eik_contrib / 1.5 in the original).
            'cap_ratio': 1 / 1.5}


def ramp_weight(cfg, epoch):
    """(trans_err_weight, rot_err_weight) at ``epoch``."""
    w = cfg['weight'] * min(max((epoch - cfg['ramp_start']) / float(cfg['ramp_len']), 0.0), 1.0)
    return w, w


def cap_scale(eik, err, weight_T, ratio):
    """Detached factor limiting the RNL's causality-weighted contribution to
    ``ratio`` x the Eikonal one, as in Shape-NTFields.  Only rescales the RNL
    term; the Eikonal term and its gradient are untouched."""
    eik_c = (eik * weight_T).sum()
    err_c = (err * weight_T).sum()
    return torch.clamp(ratio * eik_c / (err_c + 1e-12), max=1.0).detach()


def relative_normal_loss(fn, points, Xp, dtau, S_star_trans, S_star_rot, trans_n, rot_n, cfg):
    """Per-sample RNL at the band endpoint x0, goal x1 held fixed (Algorithm 1).

    fn            : the package's Function -- provides network, gradient, dim and
                    block_speeds(X, dX) -> (S_trans, S_rot), the speed its
                    Eikonal term constrains.
    Xp, dtau      : network inputs / gradients from the Eikonal pass (reused for S(q_s)).
    S_star_*      : (B,) geometry speed targets at x0.
    trans_n, rot_n: (B, 2*dim) dataset normals; only x0's halves are used.
    Returns (L_trans (B,), L_rot (B,), stats dict of floats).
    """
    d = fn.dim
    B = points.shape[0]
    q_s, q_g = points[:, :d], points[:, d:]
    q_t, q_r, valid = make_probes(q_s, trans_n[:, 0:3], rot_n[:, 3:d], cfg['eps'], cfg['mode'])
    probes = torch.cat([torch.cat([q_t, q_g], 1), torch.cat([q_r, q_g], 1)], 0)
    tau1, _, Xp1 = fn.network.out(probes)
    dtau1 = fn.gradient(tau1, Xp1)

    St0, Sr0 = fn.block_speeds(Xp[:, :d], dtau[:, :d])
    St1, _ = fn.block_speeds(Xp1[:B, :d], dtau1[:B, :d])
    _, Sr1 = fn.block_speeds(Xp1[B:, :d], dtau1[B:, :d])

    Lt, Gt, at = rnl_terms(St0, St1, S_star_trans, cfg['eps'], cfg['rho_trans'], cfg['G_max'])
    Lr, Gr, ar = rnl_terms(Sr0, Sr1, S_star_rot, cfg['eps'], cfg['rho_rot'], cfg['G_max'], valid)
    with torch.no_grad():
        stats = {'rnl_trans': Lt.mean().item(), 'rnl_rot': Lr.mean().item(),
                 'viol_trans': ((Gt > cfg['rho_trans']).float() * at).sum().item() / max(at.sum().item(), 1),
                 'viol_rot': ((Gr > cfg['rho_rot']).float() * ar).sum().item() / max(ar.sum().item(), 1),
                 'masked_rot': 1 - valid.float().mean().item()}
    return Lt, Lr, stats
