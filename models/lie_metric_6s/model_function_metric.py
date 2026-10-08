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

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

# Prior versions relied on ``torch_kdtree`` for nearest neighbour queries.
# The implementation no longer depends on that package.

torch.backends.cudnn.benchmark = True


def _hat(v):
    """(B,3) -> (B,3,3) skew-symmetric matrix [v]x."""
    z = torch.zeros_like(v[:, 0])
    return torch.stack([torch.stack([z, -v[:, 2], v[:, 1]], -1),
                        torch.stack([v[:, 2], z, -v[:, 0]], -1),
                        torch.stack([-v[:, 1], v[:, 0], z], -1)], -2)


def so3_right_jacobian_inv(phi):
    """Inverse right Jacobian of SO(3) at rotation vector ``phi`` (B,3), radians.

        Jr^{-1}(phi) = I + 1/2 [phi]x + c(theta) [phi]x^2,
        c(theta)     = 1/theta^2 - (1 + cos theta) / (2 theta sin theta)

    c is replaced by its Taylor series near 0 and theta is capped just below pi.
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
    rotation-vector input (normalized by 2*pi), in the same normalized units
    (per turn): grad_w T = Jr^{-1}(phi)^T dT/dr.  Same as models/lie_metric."""
    J = so3_right_jacobian_inv(2.0 * math.pi * rot.detach())
    return torch.einsum('bij,bi->bj', J, dT_drot)


class Function():
    def __init__(self, path, device, network, dim):

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
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def Loss(self, points, speed, beta, epoch):
        # Six-speed anisotropic Eikonal loss on SE(3).  Per endpoint, with
        # g = (dtau/dp, grad_w tau) -- the world-frame position gradient and the
        # body-frame Lie rotation gradient -- and one speed per dimension:
        #     N = sqrt(g^T M g),  M = diag(S_x^2, S_y^2, S_z^2, S_r1^2, S_r2^2, S_r3^2)
        # mapped to the loss as metric_baseline maps S|grad tau|: 1e-2*(sqrt(N)-1)^2.
        # speed: (B, 2, dim), one row per endpoint.  No normal / TD terms.
        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        diff = 0
        for e, o in enumerate((0, self.dim)):
            g = torch.cat((dtau[:, o:o+3],
                           lie_rot_grad(Xp[:, o+3:o+self.dim], dtau[:, o+3:o+self.dim])), dim=1)
            N = torch.sqrt(torch.sum(speed[:, e]**2 * g**2, dim=1) + 1e-8)
            diff = diff + 1e-2*(torch.sqrt(N) - 1)**2

        T = tau[:,0]
        loss_n = torch.sum(diff*torch.exp(-0.5*T))/points.shape[0]

        loss = beta*loss_n

        return loss, loss_n, diff

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

        #T3    = tau[:,0]**2



        #TT = LogTau * torch.sqrt(T0)

        #print(tau.shape)

        #print(T02.shape)
        #T1    = T0*torch.einsum('ij,ij->i', DT0, DT0)
        #T2    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
        
        
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
    
    def plot(self,epoch,total_train_loss,alpha,source):
        limit = 0.5#0.5
        xmin     = [-limit,-limit]
        xmax     = [limit,limit]
        spacing=limit/40.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))
        #dims_n = np.setdiff1d([0,1,2],dims)[0]

        size=81

        spacing = 2*limit/(size-1)#2.0#0.4
        X,Y  = np.meshgrid( np.arange(-limit,limit+0.1*spacing,spacing),
                            np.arange(-limit,limit+0.1*spacing,spacing))
        I,J  = np.meshgrid( np.arange(0,size,1),
                            np.arange(0,size,1))
        #print(X)
        #print(X.shape)
        #print(I.shape)
        V = np.zeros((size*size,3))
        V[:,0]=X.flatten()
        V[:,1]=Y.flatten()

        IDX = I*size+J

        F = np.zeros((2*(size-1)*(size-1),3))
        F[:(size-1)*(size-1),0]=IDX[:size-1,:size-1].flatten()
        F[:(size-1)*(size-1),1]=IDX[1:,:size-1].flatten()
        F[:(size-1)*(size-1),2]=IDX[:size-1,1:].flatten()
        F[(size-1)*(size-1):,0]=IDX[1:,1:].flatten()
        F[(size-1)*(size-1):,2]=IDX[1:,:size-1].flatten()
        F[(size-1)*(size-1):,1]=IDX[:size-1,1:].flatten()

        F=F.astype(int)

        Xsrc = source#[0]*self.dim
        #Xsrc[0] = -0.2
        #Xsrc[1] = -0.2
        #Xsrc[2] = -0.02
        #print(self.dim)
        #Xsrc=[-1.2, 0.4-0.5*np.pi, 1.4, 0.2-0.5*np.pi,-0.5,0.9]
        #-1.3, 0.6, 1.1, 0.2,-1.5,0.9
        #scale = np.pi/0.5
        
        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,0]  = X.flatten()
        XP[:,1]  = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.device)

        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        #tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S  = ss.to('cpu').data.numpy().reshape(X.shape)
        #TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,S,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,10,0.02), cmap='gist_heat', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')
        plt.close(fig)
        #print(TT.flatten().shape)
        V[:,2] = TT.flatten()#0.2*
        igl.write_triangle_mesh(self.path+"/plots_TT"+str(epoch)+".obj", V, F) 


        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,0]  = X.flatten()
        XP[:,1]  = Y.flatten()
        XP[:,2] = -0.1
        XP = Variable(Tensor(XP)).to(self.device)

        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        #tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S  = ss.to('cpu').data.numpy().reshape(X.shape)
        #TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,S,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,10,0.02), cmap='gist_heat', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path+"/tauplots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')

        plt.close(fig)

        V[:,2] = TT.flatten()
        igl.write_triangle_mesh(self.path+"/plots_TAU"+str(epoch)+".obj", V, F) 


         
