import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import nntools as nt
from utils import DenoisingStatsManager
from utils import NNRegressor


class HT(nn.Module):
    def __init__(self, n_ch=64):
        super(HT, self).__init__()
    def forward(self, x, thres):
        out=torch.clone(x)
        for i in range(x.size(0)):
            out[i] = torch.clamp(x[i],min=-thres[i],max=thres[i])
        return out

class SoftShk(nn.Module):
    def __init__(self, n_ch=64):
        super(SoftShk, self).__init__()

        self.n_ch = n_ch
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, l):
        out = self.relu1(x - l) - self.relu2(-x - l)
        return out


class unfolded_ScCP_ver2(NNRegressor):
    def __init__(self, K, F, device='cpu'):
        super(unfolded_ScCP_ver2, self).__init__()
        self.K = K
        self.F = F
        self.norm_net = 0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.K):
            self.conv.append(nn.Conv2d(in_channels = 3, out_channels = F, kernel_size = 3, padding = 1, bias = False, dtype=torch.float))
            self.conv.append(
                nn.ConvTranspose2d(in_channels = F, out_channels = 3, kernel_size = 3, padding = 1, bias = False, dtype=torch.float))
            self.conv[i * 2 + 1].weight = self.conv[i * 2].weight
            nn.init.kaiming_normal_(self.conv[i * 2].weight.data, nonlinearity = 'relu')

        x=np.ones(K)
        self.mu = nn.Parameter(torch.tensor(x,requires_grad=True,dtype=torch.float).cpu())
        # self.alpha = nn.Parameter(torch.tensor(x,requires_grad=True,dtype=torch.float).cuda())
        self.ht = HT()
        self.lip = torch.tensor(np.ones(K),requires_grad=False,dtype=torch.float).cpu()
        # apply He's initialization
        for i in range(K):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        outputdir= 'checkpoints/unfolded_ScCP_ver2/unfolded_ScCP_ver2_F64_K20_batchsize200_param_34580_data_vary'
        # outputdir= 'checkpoints/unfolded_ISTA_ver2/unfolded_ISTA_ver2_F64_K20_batchsize200_param_34560_data_vary'

        checkpoint_path = os.path.join(outputdir, "checkpoint.pth.tar")
        checkpoint = torch.load(checkpoint_path,map_location=device)
        self.load_state_dict(checkpoint['Net'])

    def lip_cal(self,x):
        K=self.K
        for i in range(K):
            tol = 1e-4
            max_iter = 10
            with torch.no_grad():
                xtmp = torch.randn_like(x)
                xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
                val = 1
                for k in range(max_iter):
                    old_val = val
                    xtmp = self.conv[2*i+1](self.conv[2*i](xtmp))
                    val = torch.linalg.norm(xtmp.flatten())
                    rel_val = torch.absolute(val - old_val) / old_val
                    if rel_val < tol:
                        break
                    xtmp = xtmp / val
            self.lip[i]= 0.99 / val
    def forward(self, z,delta=0.03):
        K = self.K
        # Initialization
        x_prev = z
        x_curr = z
        u = self.conv[0](z)
        #1st---> [K-1]-th layer
        gamma =1
        for k in range(K):
            tol = 1e-4
            max_iter = 50
            with torch.no_grad():
                xtmp = torch.randn_like(z)
                xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
                val = 1
                for i in range(max_iter):
                    old_val = val
                    xtmp = self.conv[2 * k + 1](self.conv[2 * k](xtmp))
                    val = torch.linalg.norm(xtmp.flatten())
                    rel_val = torch.absolute(val - old_val) / old_val
                    if rel_val < tol:
                        break
                    xtmp = xtmp / val
            tau = 0.99 / val

            #if k>=1:
            #    self.alpha.data[k-1] = 1/torch.sqrt(1+2*gamma*self.mu.data[k-1])
            #    self.mu.data[k] = self.alpha.data[k-1]*self.mu.data[k-1]
            #    tau = tau/self.alpha.data[k-1]
            alphak = 1/torch.sqrt(1+2*gamma*self.mu.data[k])
            u = functional.hardtanh(u+tau/self.mu[k]*self.conv[k*2]((1+alphak)*x_curr-alphak*x_prev),min_val=-delta**2,max_val=delta**2)
            x_next = torch.clamp((self.mu[k]/(self.mu[k]+1))*z+(1/(1+self.mu[k]))*x_curr- (self.mu[k]/(self.mu[k]+1))* self.conv[k*2+1](u),min=0,max=1)
            x_prev= x_curr
            x_curr= x_next

        # K-th layer
        return x_curr
    def forward_vary(self, z,delta=0.03):
        K = self.K
        # Initialization
        x_prev = z
        x_curr = z
        u = self.conv[0](z)
        #1st---> [K-1]-th layer
        gamma =1
        for k in range(K):
            tol = 1e-4
            max_iter = 50
            with torch.no_grad():
                xtmp = torch.randn_like(z)
                xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
                val = 1
                for i in range(max_iter):
                    old_val = val
                    xtmp = self.conv[2 * k + 1](self.conv[2 * k](xtmp))
                    val = torch.linalg.norm(xtmp.flatten())
                    rel_val = torch.absolute(val - old_val) / old_val
                    if rel_val < tol:
                        break
                    xtmp = xtmp / val
            tau = 0.99 / val

            #if k>=1:
            #    self.alpha.data[k-1] = 1/torch.sqrt(1+2*gamma*self.mu.data[k-1])
            #    self.mu.data[k] = self.alpha.data[k-1]*self.mu.data[k-1]
            #    tau = tau/self.alpha.data[k-1]
            alphak = 1/torch.sqrt(1+2*gamma*self.mu.data[k])
            u = self.ht(u+tau/self.mu[k]*self.conv[k*2]((1+alphak)*x_curr-alphak*x_prev),thres=delta**2)
            x_next = torch.clamp((self.mu[k]/(self.mu[k]+1))*z+(1/(1+self.mu[k]))*x_curr- (self.mu[k]/(self.mu[k]+1))* self.conv[k*2+1](u),min=0,max=1)
            x_prev= x_curr
            x_curr= x_next
        # K-th layer
        return x_curr
    def forward_eval(self, z,delta=0.03,dualvar=None):
        K = self.K
        # Initialization
        x_prev = z
        x_curr = z

        if dualvar==None:
            u = self.conv[0](z)
        else:
            u = dualvar
        #1st---> [K-1]-th layer
        gamma =1
        for k in range(K):
            #if k>=1:
            #    self.alpha.data[k-1] = 1/torch.sqrt(1+2*gamma*self.mu.data[k-1])
            #    self.mu.data[k] = self.alpha.data[k-1]*self.mu.data[k-1]
            #    tau = tau/self.alpha.data[k-1]
            alphak = 1/torch.sqrt(1+2*gamma*self.mu.data[k])
            u = functional.hardtanh(u+self.lip[k]/self.mu[k]*self.conv[k*2]((1+alphak)*x_curr-alphak*x_prev),min_val=-delta**2,max_val=delta**2)
            x_next = torch.clamp((self.mu[k]/(self.mu[k]+1))*z+(1/(1+self.mu[k]))*x_curr- (self.mu[k]/(self.mu[k]+1))* self.conv[k*2+1](u),min=0,max=1)
            x_prev= x_curr
            x_curr= x_next
        # K-th layer
        return x_curr

