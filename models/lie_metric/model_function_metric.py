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

    c is replaced by its Taylor series near 0 and theta is capped just below pi
    (the cut locus, where the rotation-vector chart is singular).
    """
    theta = phi.norm(dim=-1)
    small = theta < 1e-2
    th = torch.where(small, torch.ones_like(theta), theta).clamp(max=math.pi - 1e-3)
    c = 1.0 / th**2 - (1.0 + torch.cos(th)) / (2.0 * th * torch.sin(th))
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


def aniso_residual(dT_dp, grad_w, St, Sr, eps=1e-8):
    """N(g) = sqrt(St^2 |grad_v T|^2 + Sr^2 |grad_w T|^2), target 1 (eq. 11).
    |grad_v T| = |dT/dp| since the body translation block is R^T dT/dp."""
    return torch.sqrt(St**2 * (dT_dp**2).sum(-1)
                      + Sr**2 * (grad_w**2).sum(-1) + eps)


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
        self.residual = 'N'

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
        # Block-anisotropic Eikonal loss on SE(3) (anisotropic-speed notes,
        # eqs. 3/10/11).  Per endpoint, with the body-frame Lie gradient
        # (grad_v tau, grad_w tau):
        #     N = sqrt(S_dist^2 |grad_v tau|^2 + S_angle^2 |grad_w tau|^2),  (N-1)^2
        # The translation block is the plain position gradient (only its norm
        # enters); the rotation block maps dtau/d(rotvec/2pi) through Jr^{-1}.
        # No normal / TD terms.
        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        half_dim = 3
        diff_4 = 0
        for o, e in ((0, 0), (self.dim, 1)):
            grad_v = dtau[:, o:o+half_dim]
            grad_w = lie_rot_grad(Xp[:, o+half_dim:o+self.dim], dtau[:, o+half_dim:o+self.dim])
            N = aniso_residual(grad_v, grad_w, speed_dist[:, e], speed_angle[:, e])
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


         
