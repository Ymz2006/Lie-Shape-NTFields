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
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def block_speeds(self, X, dX):
        """Network-implied per-block speeds of one endpoint, as the blocked
        Eikonal term constrains them: S_c = 1 / |grad_c tau| on the plain
        coordinate gradient.  X, dX: (B, dim) inputs / dtau of that endpoint."""
        S_trans = 1.0 / torch.sqrt((dX[:, :3]**2).sum(-1) + 1e-8)
        S_rot = 1.0 / torch.sqrt((dX[:, 3:self.dim]**2).sum(-1) + 1e-8)
        return S_trans, S_rot

    def Loss(self, points, Yobs, normal, beta, gamma, epoch, speed_dist, speed_angle,
             trans_n=None, rot_n=None):
        # Blocked Eikonal loss: the translation and rotation blocks of grad(tau)
        # each satisfy their own Eikonal equation against their own speed,
        #     (S_dist |grad_xyz tau| - 1)^2 + (S_angle |grad_rot tau| - 1)^2
        # per endpoint, on plain coordinate gradients.  No normal / TD terms.
        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        half_dim = 3
        diff_4 = 0
        for o, e in ((0, 0), (self.dim, 1)):
            for block, S in ((dtau[:, o:o+half_dim], speed_dist[:, e]),
                             (dtau[:, o+half_dim:o+self.dim], speed_angle[:, e])):
                diff_4 = diff_4 + (torch.sqrt((block**2).sum(-1) + 1e-8) * S - 1)**2

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


         
