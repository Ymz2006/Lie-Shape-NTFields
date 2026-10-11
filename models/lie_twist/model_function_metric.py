import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_mlp as db
#from models import model_network_one as model_network
import igl 
import copy

from models import rnl

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

# Prior versions relied on ``torch_kdtree`` for nearest neighbour queries.
# The implementation no longer depends on that package.

# Optional Weights & Biases logging. Guarded by ``wandb.run is not None`` so it
# is a no-op unless a trainer started a run (see train/wandb_utils.py).
try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cudnn.benchmark = True


def _hat(v):
    """(B,3) -> (B,3,3) skew-symmetric matrix [v]x."""
    z = torch.zeros_like(v[:, 0])
    return torch.stack([torch.stack([z, -v[:, 2], v[:, 1]], -1),
                        torch.stack([v[:, 2], z, -v[:, 0]], -1),
                        torch.stack([-v[:, 1], v[:, 0], z], -1)], -2)


def so3_right_jacobian_inv(phi):
    """Inverse right Jacobian of SO(3) at rotation vector ``phi`` (B,3), radians.

        log(exp(phi) exp(d)) = phi + Jr^{-1}(phi) d + O(|d|^2)
        Jr^{-1}(phi) = I + 1/2 [phi]x + c(theta) [phi]x^2,
        c(theta)     = 1/theta^2 - (1 + cos theta) / (2 theta sin theta)

    Written as c = 1/theta^2 - cot(theta/2) / (2 theta), which is finite through
    theta = pi, so it also holds for rotation vectors past the cut locus (theta in
    (pi, 2pi), e.g. MPPI states fed to a matrix-input network); theta is capped
    just below 2pi, where the chart is singular.  Taylor series near 0.
    """
    theta = phi.norm(dim=-1)
    small = theta < 1e-2
    th = torch.where(small, torch.ones_like(theta), theta).clamp(max=2.0 * math.pi - 1e-3)
    c = 1.0 / th**2 - 1.0 / (2.0 * th * torch.tan(0.5 * th))
    c = torch.where(small, 1.0 / 12.0 + theta**2 / 720.0, c)
    K = _hat(phi)
    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand_as(K)
    return I + 0.5 * K + c[:, None, None] * (K @ K)


def lie_rot_grad(rot, dT_drot):
    """Body-frame rotational Lie gradient (E4 T, E5 T, E6 T) for the
    rotation-vector input encoding, i.e. eq. (9) of the anisotropic-speed notes
    evaluated in closed form.

    ``rot`` is the network's rotation input -- a rotation vector normalized by
    2*pi -- and ``dT_drot`` the autograd gradient w.r.t. it, both (B,3).  The
    body perturbation R <- R exp([delta]x) is taken in the same normalized units
    (delta measured in turns, not radians), so the result is on the same scale as
    the existing ``speed_angle`` targets.  With r = phi/(2 pi) and
    phi(delta) = log(exp(phi) exp(2 pi delta)):

        dr/ddelta = Jr^{-1}(phi)   =>   grad_w T = Jr^{-1}(phi)^T dT/dr

    Jr^{-1} only depends on the input, so it carries no network gradient; the
    create_graph'd dT_drot is what the second-order loss differentiates through.
    """
    J = so3_right_jacobian_inv(2.0 * math.pi * rot.detach())
    return torch.einsum('bij,bi->bj', J, dT_drot)


# ---------------------------------------------------------------------------
# Riemannian Eikonal on SE(3) from body-twist optimal control
#
#   dynamics   gdot = g xi^,  xi = (v, w) in se(3) (body twist, translation first)
#   control    xi^T G(g) xi <= 1
#   cost       T(g, g_g) = min int 1/S(g) dt
#   HJB        0 = min_{xi^T G xi <= 1} [1/S + p_b . xi]  =>  sqrt(p_b^T G^{-1} p_b) = 1/S
#   control    xi* = -G^{-1} p_b / sqrt(p_b^T G^{-1} p_b)
#
# p_b is the left-trivialized gradient, p_b . xi = d/de T(g exp(e xi^)).  In the
# network's chart chi = (p, r = phi/2pi) the body twist is xi = B(chi) chidot with
# B = blockdiag(R^T, 2pi Jr(phi)), so p_b = B^{-T} dT/dchi = (R^T dT/dp, lie_rot_grad).
#
# The metric keeps lie_metric's clearance speeds and replaces its identity
# Lie-algebra metric with the shape's own twist metric G0:
#
#   G(g) = D^{-1} G0 D^{-1},  D = diag(St I3, Sr I3),  S = 1
#   =>  N = sqrt((D p_b)^T G0^{-1} (D p_b)) = 1
#
# xi^T G0 xi is the mean squared velocity of the shape's surface points under xi,
# so the time of a motion is the RMS distance the body's surface sweeps, with the
# translational part slowed by St and the rotational part by Sr.  G0 = I gives
# lie_metric exactly.  G0 is anisotropic in w (rotating about a long axis sweeps
# less) and couples v and w when the body origin (bbox centre) is off the centroid.
# ---------------------------------------------------------------------------
def so3_exp(phi):
    """Rotation matrix of rotation vector ``phi`` (B,3), radians.  Taylor-safe at 0."""
    theta = phi.norm(dim=-1)
    small = theta < 1e-4
    th = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(th)) / th**2)
    K = _hat(phi)
    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand_as(K)
    return I + a[:, None, None] * K + b[:, None, None] * (K @ K)


def body_twist_grad(X, dX):
    """Left-trivialized gradient p_b = B^{-T} dT/dchi of one endpoint, (B,6).

    X, dX: (B,6) chart coordinates (p, rotvec/2pi) and dT w.r.t. them.  The
    translation block is the body-frame R^T dT/dp; the rotation block is
    lie_rot_grad (per turn, like the speed_angle targets).  R carries no network
    gradient, so the second-order loss differentiates through dX only.
    """
    R = so3_exp(2.0 * math.pi * X[:, 3:6].detach())
    grad_v = torch.einsum('bji,bj->bi', R, dX[:, :3])
    grad_w = lie_rot_grad(X[:, 3:6], dX[:, 3:6])
    return torch.cat([grad_v, grad_w], dim=-1)


def shape_twist_metric(V, F, rot_norm=2.0 * math.pi):
    """Twist metric G0 (6,6) of a triangle mesh: xi^T G0 xi = mean over the surface
    (area-weighted) of |v + (rot_norm w) x x|^2, with w per turn when rot_norm = 2pi.

    With c = mean x and J = mean(|x|^2 I - x x^T) about the body origin,
        G0 = [[I, -k [c]x], [k [c]x, k^2 J]],  k = rot_norm.
    Exact per-triangle moments: int x dA = A s/3, int x x^T dA = A/12 (sum_i v_i v_i^T
    + s s^T), s = a + b + c.
    """
    tri = np.asarray(V, dtype=np.float64)[np.asarray(F)]          # (m,3,3)
    s = tri.sum(1)                                                # (m,3)
    A = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = A.sum()
    c = (A[:, None] * s).sum(0) / 3.0 / area
    M2 = (A[:, None, None] / 12.0 * (np.einsum('mki,mkj->mij', tri, tri)
                                     + s[:, :, None] * s[:, None, :])).sum(0) / area
    J = np.trace(M2) * np.eye(3) - M2
    Kc = _hat(torch.tensor(c[None])).numpy()[0]
    G0 = np.zeros((6, 6))
    G0[:3, :3] = np.eye(3)
    G0[:3, 3:] = -rot_norm * Kc
    G0[3:, :3] = rot_norm * Kc
    G0[3:, 3:] = rot_norm**2 * J
    return G0


def load_shape_twist_metric(data_path, kind='shape'):
    """G0 for the dataset's shape, in the same normalized units as the samples:
    vertices centred on the bbox centre (the body origin) and divided by the env
    scale, exactly as dataprocessing/preprocess_obj.py places the shape.

    kind 'shape' is the RMS (surface-mean) twist metric.  'shape_max' rescales each
    body rotation axis so its diagonal entry is the squared WORST-CASE point speed
    per turn about that axis, (rot_norm max_x |e_a x x|)^2 instead of the mean, keeping
    G0's coupling structure (G0 -> S G0 S, S = diag(I, max_a / rms_a)): collisions are
    caused by the fastest-moving point, which the mean undercounts most for rotation
    about a thin axis."""
    import os
    import json
    from dataprocessing.preprocess_obj import load_obj   # lazy: pulls in plotly
    meta = json.load(open(os.path.join(data_path, 'meta.json')))
    path = meta['shape_obj']
    if not os.path.exists(path):
        path = os.path.join('./datasets/3dshape', os.path.basename(path))
    V, F, _ = load_obj(path)
    V = (V - 0.5 * (V.min(0) + V.max(0))) / float(meta['env_scale']) * float(meta['shape_scale'])
    k = float(meta.get('rot_norm', 2.0 * math.pi))
    G0 = shape_twist_metric(V, F, k)
    if kind == 'shape_max':
        mx = np.array([k * np.linalg.norm(np.cross(np.eye(3)[a], V), axis=1).max() for a in range(3)])
        Sc = np.diag(np.concatenate([np.ones(3), mx / np.sqrt(np.diag(G0)[3:])]))
        G0 = Sc @ G0 @ Sc
    return G0


def twist_residual(p_b, St, Sr, G0_inv, eps=1e-8):
    """N = sqrt(p_b^T G(g)^{-1} p_b) with G^{-1} = D G0^{-1} D, target 1."""
    q = torch.cat([St[:, None] * p_b[:, :3], Sr[:, None] * p_b[:, 3:]], dim=-1)
    return torch.sqrt(torch.einsum('bi,ij,bj->b', q, G0_inv, q) + eps)


def optimal_twist(p_b, St, Sr, G0_inv, eps=1e-8):
    """xi* = -G^{-1} p_b / sqrt(p_b^T G^{-1} p_b): the unit-G body twist that
    decreases T fastest (image section 4).  (B,6), translation first."""
    D = torch.cat([St[:, None].expand(-1, 3), Sr[:, None].expand(-1, 3)], dim=-1)
    Ginv_p = D * ((D * p_b) @ G0_inv.T)
    return -Ginv_p / torch.sqrt((p_b * Ginv_p).sum(-1, keepdim=True) + eps)


class Function():
    def __init__(self, path, device, network, dim, data_path=None):

        # ======================= JSON Template =======================
        self.path = path
        self.device = device
        self.dim = dim

        self.network = network

        # Pass the JSON information
        #self.Params['Device'] = device

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        #input_file = "datasets/gibson/Cabin/mesh_z_up_scaled.off"
        #self.kdtree, self.v_obs, self.n_obs = self.pc_kdtree(input_file)

        self.alpha = 1.025
        limit = 0.5
        self.margin = limit/15.0
        self.offset = self.margin/10.0

        # Environment boundary points (for the 2-D shape task) -- used only by
        # plot() to overlay the obstacles.  Optional: stays None if absent.
        self.env = None
        if data_path is not None:
            import os
            env_file = os.path.join(data_path, 'env.npy')
            if os.path.exists(env_file):
                self.env = np.load(env_file)

        # Relative normal loss config (models/rnl.py); mode 'none' leaves the
        # loss untouched.  Model.train() overrides it from Params['Training']['RNL'].
        self.rnl = dict(rnl.DEFAULTS)
        self.rnl_stats = {}
        # Residual-to-loss mapping: 'N' = (N-1)^2, 'sqrtN' = (sqrt(N)-1)^2 (the
        # metric_baseline mapping).  Model.train() overrides it from
        # Params['Training']['Residual'].
        self.residual = 'sqrtN'
        # Speed form of the HJB, sqrt(p^T G^{-1} p) = 1/S; Model.train()
        # overrides both from Params['Training']:
        #   'block'    G = D^{-1} G0 D^{-1}, D = diag(S_dist I, S_angle I), S = 1
        #   'dist'     S = S_dist, G = G0 (constant)  -- the image's scalar speed
        #   'combined' S = metric_baseline's combined speed, G = G0 (constant)
        # With a constant G the optimal-twist direction -G0^{-1} p_b depends on
        # tau alone.  grad: 'lie' = left-trivialized p_b, 'chart' = dtau/dchi
        # (coordinate gradient; the non-Lie control).
        self.speed_form = 'block'
        self.grad = 'lie'

        # Shape twist metric G0 (6,6); set_twist_metric() is called by
        # Model.train() / Model.load().  Identity until then (= lie_metric).
        self.set_twist_metric(np.eye(6))

    def set_twist_metric(self, G0):
        G0 = torch.as_tensor(np.asarray(G0), dtype=torch.float32)
        self.G0 = G0.to(self.device)
        self.G0_inv = torch.linalg.inv(G0.double()).float().to(self.device)

    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def block_speeds(self, X, dX):
        """Network-implied per-block speeds of one endpoint, on the same body-frame
        Lie gradient the anisotropic Eikonal term uses: S_trans = 1/|grad_v tau|,
        S_rot = 1/|grad_w tau|.  X, dX: (B, dim) inputs / dtau of that endpoint."""
        grad_w = lie_rot_grad(X[:, 3:self.dim], dX[:, 3:self.dim])
        S_trans = 1.0 / torch.sqrt((dX[:, :3]**2).sum(-1) + 1e-8)
        S_rot = 1.0 / torch.sqrt((grad_w**2).sum(-1) + 1e-8)
        return S_trans, S_rot

    def Loss(self, points, Yobs, normal, beta, gamma, epoch, speed_dist, speed_angle,
             trans_n=None, rot_n=None):
        # Riemannian Eikonal loss on SE(3) from body-twist optimal control (see
        # the header above twist_residual).  Per endpoint, with the
        # left-trivialized gradient p_b = (R^T dtau/dp, Jr^{-T} dtau/dr):
        #     N = sqrt((D p_b)^T G0^{-1} (D p_b)),  D = diag(S_dist I, S_angle I)
        # mapped (sqrt(N)-1)^2 or (N-1)^2.  No normal / TD terms.
        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        diff_4 = 0
        for o, e in ((0, 0), (self.dim, 1)):
            if self.grad == 'chart':
                p_b = dtau[:, o:o+self.dim]
            else:
                p_b = body_twist_grad(Xp[:, o:o+self.dim], dtau[:, o:o+self.dim])
            if self.speed_form == 'block':
                N = twist_residual(p_b, speed_dist[:, e], speed_angle[:, e], self.G0_inv)
            else:
                S = speed_dist[:, e] if self.speed_form == 'dist' else Yobs[:, e]
                ones = torch.ones_like(S)
                N = S * twist_residual(p_b, ones, ones, self.G0_inv)
            if self.residual == 'sqrtN':
                diff_4 = diff_4 + (torch.sqrt(N) - 1)**2
            else:
                diff_4 = diff_4 + (N - 1)**2

        loss_weight = 1e-2
        diff_4 = diff_4 * loss_weight
        T = tau[:, 0]
        weight_T = torch.exp(-0.5 * T)


        
        trans_err_weight, rot_err_weight = rnl.ramp_weight(self.rnl, epoch)
        if self.rnl['mode'] != 'none' and trans_err_weight + rot_err_weight > 0:
            # RNL on the band endpoint x0 only, same causality weight as Eikonal,
            # capped relative to the Eikonal term as in Shape-NTFields.
            L_trans, L_rot, self.rnl_stats = rnl.relative_normal_loss(
                self, points, Xp, dtau, speed_dist[:, 0], speed_angle[:, 0],
                trans_n, rot_n, self.rnl)
            err = trans_err_weight * L_trans + rot_err_weight * L_rot
            scale = rnl.cap_scale(diff_4, err, weight_T, self.rnl['cap_ratio'])
            self.rnl_stats['err_weight'] = trans_err_weight
            self.rnl_stats['cap_scale'] = scale.item()
            diff_4 = diff_4 + scale * err
        loss_n = torch.sum(diff_4 * weight_T) / Yobs.shape[0]

        return loss_n, loss_n, diff_4

    def TravelTimes(self, Xp):
     
        tau, w, coords = self.network.out(Xp)        

        TT = tau[:,0] #* torch.sqrt(T0)
            
        return TT

    def Speed(self, Xp):

   

        Xp = Xp.to(torch.device(self.device))

        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)
        
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.einsum('ij,ij->i', D, D)

        #DT0 = dtau[:,self.dim:]
        DT0 = dtau[:,:self.dim]

        
        
        S = torch.einsum('ij,ij->i', DT0, DT0)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau#, T0#, T1, T2, T3
        return Ypred

    def OptimalTwist(self, Xp, St=None, Sr=None):
        """Optimal body twist xi* (B,6) at the start slice of Xp toward its goal
        slice, unit length in G(g).  St / Sr default to 1 (free space)."""
        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp, create_graph=False)
        p_b = body_twist_grad(Xp[:, :self.dim], dtau[:, :self.dim])
        ones = torch.ones_like(p_b[:, 0])
        St = ones if St is None else St
        Sr = ones if Sr is None else Sr
        return optimal_twist(p_b.detach(), St, Sr, self.G0_inv)

    def Gradient(self, Xp):
        #Xp = Xp.to(torch.device(self.device))
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)
        #print(Xp.shape)
        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D)).view(-1,1)

        #A = T0*dtau[:,:self.dim]
        #B = tau/T0*D
        

        Ypred0 = -dtau[:,:self.dim]#-A+B
        #print(Ypred0.shape)
        Spred0 = torch.norm(Ypred0,dim=1).view(-1,1)
        Ypred0 = 1/Spred0**2*Ypred0

        Ypred1 = -dtau[:,self.dim:]#-A-B
        Spred1 = torch.norm(Ypred1,dim=1).view(-1,1)

        Ypred1 = 1/Spred1**2*Ypred1

        #print(Ypred0.shape)
        #print(Ypred1.shape)
        
        return torch.cat((Ypred0, Ypred1),dim=1)
    
    def plot(self, epoch, total_train_loss, alpha, source):
        # Travel-time / speed field over an (x, y) grid, with the rest of the
        # configuration (theta for the 2-D shape task) held fixed at `source`.
        limit = 0.5
        size = 81
        spacing = 2 * limit / (size - 1)
        X, Y = np.meshgrid(np.arange(-limit, limit + 0.1 * spacing, spacing),
                           np.arange(-limit, limit + 0.1 * spacing, spacing))

        Xsrc = source  # goal configuration -- T is measured from here

        XP = np.zeros((len(X.flatten()), 2 * self.dim))
        XP[:, :self.dim] = Xsrc
        XP[:, self.dim:] = Xsrc
        XP[:, 0] = X.flatten()
        XP[:, 1] = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.device)

        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)

        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S = ss.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X, Y, S, vmin=0, vmax=1)
        ax.contour(X, Y, TT, np.arange(0, 10, 0.02), cmap='gist_heat', linewidths=0.5)
        if self.env is not None:
            ax.scatter(self.env[:, 0], self.env[:, 1], c='black', s=2, zorder=3)
        ax.scatter([Xsrc[0]], [Xsrc[1]], c='red', s=40, zorder=4, label='source')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right', fontsize=7)
        plt.colorbar(quad1, ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path + "/plots" + str(epoch) + "_" + str(alpha) + "_"
                    + str(round(total_train_loss, 4)) + "_0.jpg", bbox_inches='tight')
        plt.close(fig)


         
