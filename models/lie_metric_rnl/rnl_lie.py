"""Relative Normal Loss (RNL) for the anisotropic Lie Eikonal of lie_metric.

Shape-NTFields eqs. (6)-(7) with the Lie-group probe curves of
``RNL_Lie_Group_Probes.pdf`` (probes from ``models/rnl.py::make_probes``), and a
speed function S_c that is the one lie_metric's Eikonal term constrains.

Why not the per-block speeds 1/|grad_v tau|, 1/|grad_w tau|
-----------------------------------------------------------
The anisotropic Eikonal pins only the combined norm
    N_q(p) = sqrt(St*(q)^2 |grad_v tau|^2 + Sr*(q)^2 |grad_w tau|^2) = 1,
so 1/|grad_v tau| and 1/|grad_w tau| are free (each only bounded below by its
target).  RNL on them (models/rnl.py with lie_metric) pushes on quantities the
Eikonal term never constrains; it failed at every setting (research tree,
lie_metric.rnl*).

'aniso' speed (default)
-----------------------
The network-implied speeds that the Eikonal term does constrain are the
targets scaled by the residual, S_c(q0) = Sc*(q0) / N_q0(p(q0)): the Eikonal
residual is zero exactly when (S_t, S_r) = (St*, Sr*).  At a probe q the metric
is frozen at the band point q0:
    S_t(q) = St*(q0) / N_q0(p(q)),   S_r(q) = Sr*(q0) / N_q0(p(q)).
When p is translation-dominated S_t(q) = 1/|grad_v tau(q)|, the blocked
(Shape-NTFields) speed; when it is rotation-dominated, S_r(q) = 1/|grad_w tau(q)|.

Convergence
-----------
* balance: the RNL gradient is 56-70x the Eikonal one at the start of the ramp,
  at cosine -0.6 to it.  'value' (default, Shape-NTFields) caps the RNL loss
  value at cap_ratio x the Eikonal one, which lets the RNL dominate the update:
  switched on abruptly or early in training it stalls the accept / reject loop
  (380 rejected retries of one epoch), but ramped in from a trained network
  (ramp_start 1000) it trained in 12 of 12 runs and plans best, at a 25-40%
  higher Eikonal loss.  'grad' removes the RNL component along -grad L_eik and
  caps the rest at cap_ratio x |grad L_eik| (balanced_grads): it never stalls and
  keeps the Eikonal loss and path lengths of lie_metric, for a smaller gain.
  Epochs are accepted on the Eikonal loss alone in both cases.
* anchor: in eq. (6) G = (S(q_probe) - S(q0)) / eps also differentiates
  S(q0), and the RNL pushes it *up* (to steepen G) while the Eikonal term pins
  it to S*(q0).  'detach' stops that gradient: the RNL only lowers the speed
  at the probe.
* threshold: with constant rho, G <= rho is infeasible whenever
  S(q0) < -rho * eps (rho_trans = -20, eps = 0.02: S(q0) < 0.4, a large part of
  the band), and the penalty then pushes forever.  'analytic' replaces rho by
  the target field's own finite-difference slope along the probe, computed from
  the stored clearances: the translation probe lowers the clearance by eps
  (s_d -> s_d - eps / margin), the Lie rotation probe turns the body by
  2 pi eps rad about the axis that closes the min angle (s_a -> s_a - 2 eps /
  angle_max).  The constraint G <= G* then reads S(q_probe) <= S(q0) + (S*(probe)
  - S*(q0)) and is always satisfiable.  'feasible' keeps rho but never asks for
  more than a (1 - kappa) fraction of S(q0) to survive.
"""
import torch

from models.rnl import make_probes, cap_scale

# Clearance -> speed-target warp, exactly as the training loop builds speed_dist /
# speed_angle from the stored clearances (models/lie_metric_rnl/model_train_metric.py).
MARGIN = 0.05            # meta.json margin: speed_dists = clip(dist / MARGIN, OFFSET / MARGIN, 1)
OFFSET = 0.001
ANGLE_MAX = 1.0 / 6.0    # speed_angles (= angle / pi) saturates at 30 deg
ANGLE_MIN = 1e-4


def warp(s):
    """Polynomial clearance -> speed warp of the training loop, floored at 1e-3."""
    return torch.clamp(s ** 2 * (2 - s) ** 2, min=0.001)


def angle_clearance(speed_angles):
    """Stored speed_angles (angle / pi) -> normalized clearance s_a in [ANGLE_MIN/ANGLE_MAX, 1]."""
    return torch.clamp(speed_angles / ANGLE_MAX, min=ANGLE_MIN / ANGLE_MAX, max=1)


DEFAULTS = {'mode': 'lie',           # probes: 'none' | 'orig' (coordinate lines) | 'lie' (group curves)
            'speed': 'aniso',        # 'aniso' (Eikonal-consistent, above) | 'block' (1/|grad_block tau|)
            'anchor': 'detach',      # S(q0) in eq. (6): 'net' | 'detach' | 'target' (= S*(q0))
            'rho_mode': 'analytic',  # 'const' | 'feasible' | 'analytic' (see module doc)
            'kappa': 0.9,            # 'feasible': S(q_probe) >= (1 - kappa) S(q0) is never asked for
            'eps': 0.02,             # probe size, normalized units (the rotation probe turns 2 pi eps rad)
            'rho_trans': -1 / 0.05,  # Shape-NTFields rate_trans / rate_rot
            'rho_rot': -1 / (1 / 6),
            'G_max': 10.0,           # Shape-NTFields dsdn_clip
            'weight': 1e-4,          # peak err weight, ramped in linearly
            'ramp_start': 1000,
            'ramp_len': 1500,
            'cap_ratio': 1 / 1.5,    # RNL <= cap_ratio x Eikonal: loss value ('value') or gradient norm ('grad')
            'balance': 'value',      # 'value' (Shape-NTFields loss cap) | 'grad' (balanced_grads) |
                                     # 'norm' (balanced_grads without the projection) | 'none'
            'backoff': 0.0}          # > 0: RNL weight x backoff on every rejected epoch, regrown x1.05 per accept


def balanced_grads(params, loss_eik, loss_rnl, ratio, project=True):
    """Parameter gradient of loss_eik + loss_rnl with the RNL kept from fighting
    the Eikonal term: the component of g_rnl along -g_eik is removed (PCGrad), and
    what is left is scaled to at most ``ratio`` x |g_eik|.  To first order an
    update then never raises the Eikonal loss because of the RNL, which is what
    stalls the accept / reject loop otherwise.  Returns (grads, stats)."""
    gE = torch.autograd.grad(loss_eik, params, retain_graph=True, allow_unused=True)
    gR = torch.autograd.grad(loss_rnl, params, allow_unused=True)
    gE = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gE)])
    gR = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gR)])
    nE2 = (gE * gE).sum()
    dot = (gE * gR).sum()
    cos = (dot / (nE2.sqrt() * gR.norm() + 1e-30)).item()
    if project and dot < 0:
        gR = gR - dot / (nE2 + 1e-30) * gE
    nR = gR.norm()
    scale = torch.clamp(ratio * nE2.sqrt() / (nR + 1e-30), max=1.0)
    g = gE + scale * gR
    out, o = [], 0
    for p in params:
        out.append(g[o:o + p.numel()].view_as(p))
        o += p.numel()
    return out, {'g_cos': cos, 'g_ratio': (nR / (nE2.sqrt() + 1e-30)).item(), 'g_scale': scale.item()}


def ramp_frac(cfg, epoch):
    """Linear ramp in [0, 1].  It scales the RNL weight and, under 'grad' / 'norm'
    balancing (where the norm cap binds almost at once), the cap ratio too."""
    return min(max((epoch - cfg['ramp_start']) / float(max(cfg['ramp_len'], 1)), 0.0), 1.0)


def speeds(fn, X, dX, St0, Sr0, kind):
    """(S_trans, S_rot) of one endpoint.  X, dX: (B, dim) inputs / dtau of that
    endpoint; St0, Sr0: (B,) speed targets at the band point q0 (the frozen metric)."""
    gv2, gw2 = fn.block_sq_norms(X, dX)
    if kind == 'block':
        return 1.0 / torch.sqrt(gv2 + 1e-8), 1.0 / torch.sqrt(gw2 + 1e-8)
    assert kind == 'aniso', kind
    N0 = torch.sqrt(St0 ** 2 * gv2 + Sr0 ** 2 * gw2 + 1e-8)
    return St0 / N0, Sr0 / N0


def thresholds(cfg, S_anchor_t, S_anchor_r, s_d, s_a):
    """Per-sample slope thresholds (rho_t, rho_r), each (B,)."""
    eps = cfg['eps']
    if cfg['rho_mode'] == 'const':
        return torch.full_like(s_d, cfg['rho_trans']), torch.full_like(s_a, cfg['rho_rot'])
    if cfg['rho_mode'] == 'feasible':
        k = cfg['kappa']
        return (torch.clamp(-k * S_anchor_t.detach() / eps, min=cfg['rho_trans']),
                torch.clamp(-k * S_anchor_r.detach() / eps, min=cfg['rho_rot']))
    assert cfg['rho_mode'] == 'analytic', cfg['rho_mode']
    s_d1 = torch.clamp(s_d - eps / MARGIN, min=OFFSET / MARGIN)
    s_a1 = torch.clamp(s_a - 2 * eps / ANGLE_MAX, min=ANGLE_MIN / ANGLE_MAX)
    return (warp(s_d1) - warp(s_d)) / eps, (warp(s_a1) - warp(s_a)) / eps


def penalty(S0, S1, rho, G_max, active, eps):
    """Eq. (7): (rho - clip(G, rho, G_max))^2 on in-band samples, G = (S1 - S0) / eps."""
    G = (S1 - S0) / eps
    L = (rho - torch.maximum(torch.minimum(G, torch.full_like(G, G_max)), rho)) ** 2
    return active * L, G


def relative_normal_loss(fn, points, Xp, dtau, St0, Sr0, s_d, s_a, trans_n, rot_n, cfg):
    """Per-sample RNL at the band endpoint x0, goal x1 held fixed.

    Xp, dtau : network inputs / gradients of the Eikonal pass (S(q0) is reused).
    St0, Sr0 : (B,) warped speed targets at x0 (the Eikonal's speed_dist / speed_angle).
    s_d, s_a : (B,) normalized clearances at x0 before the warp (analytic thresholds).
    trans_n, rot_n : (B, 2*dim) dataset normals; only x0's halves are used.
    Returns (L_trans (B,), L_rot (B,), stats dict of floats).
    """
    d = fn.dim
    B = points.shape[0]
    eps = cfg['eps']
    q_s, q_g = points[:, :d], points[:, d:]
    q_t, q_r, valid = make_probes(q_s, trans_n[:, 0:3], rot_n[:, 3:d], eps, cfg['mode'])
    probes = torch.cat([torch.cat([q_t, q_g], 1), torch.cat([q_r, q_g], 1)], 0)
    tau1, _, Xp1 = fn.network.out(probes)
    dtau1 = fn.gradient(tau1, Xp1)

    St_q0, Sr_q0 = speeds(fn, Xp[:, :d], dtau[:, :d], St0, Sr0, cfg['speed'])
    St_pr, _ = speeds(fn, Xp1[:B, :d], dtau1[:B, :d], St0, Sr0, cfg['speed'])
    _, Sr_pr = speeds(fn, Xp1[B:, :d], dtau1[B:, :d], St0, Sr0, cfg['speed'])

    if cfg['anchor'] == 'target':
        At, Ar = St0, Sr0
    elif cfg['anchor'] == 'detach':
        At, Ar = St_q0.detach(), Sr_q0.detach()
    else:
        assert cfg['anchor'] == 'net', cfg['anchor']
        At, Ar = St_q0, Sr_q0
    rho_t, rho_r = thresholds(cfg, At, Ar, s_d, s_a)

    act_t = (St0 < 1).to(St0.dtype)
    act_r = (Sr0 < 1).to(Sr0.dtype) * valid.to(Sr0.dtype)
    Lt, Gt = penalty(At, St_pr, rho_t, cfg['G_max'], act_t, eps)
    Lr, Gr = penalty(Ar, Sr_pr, rho_r, cfg['G_max'], act_r, eps)
    with torch.no_grad():
        nt, nr = max(act_t.sum().item(), 1), max(act_r.sum().item(), 1)
        stats = {'rnl_t': Lt.mean().item(), 'rnl_r': Lr.mean().item(),
                 'viol_t': ((Gt > rho_t).float() * act_t).sum().item() / nt,
                 'viol_r': ((Gr > rho_r).float() * act_r).sum().item() / nr,
                 'G_t': (Gt * act_t).sum().item() / nt, 'G_r': (Gr * act_r).sum().item() / nr,
                 'rho_t': (rho_t * act_t).sum().item() / nt, 'rho_r': (rho_r * act_r).sum().item() / nr,
                 'masked_r': 1 - valid.float().mean().item()}
    return Lt, Lr, stats
