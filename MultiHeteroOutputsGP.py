import torch
from torch import nn, optim
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.utils.cholesky import psd_safe_cholesky

from numpy import linalg as la
import numpy as np

class ELBO_HeteroOutputsGP(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,q_u,p_u,q_F,Y,HeteroLikehood):
        "We need to compute \sum_{d=1}^D \sum_{n=1}^N E_qfd[g] - KL(q_u||p_u)"
        "D: number of heterogeneous outputs"
        "J: total number of latent functions among all D outputs"
        "Q: number of latent GPs uq"
        "Y: this is upper case to indicate a list with all outputs, i.e., Y = [y1,y2,...,yD]"

        KL = compute_KL(q_u,p_u)
        # Variational Expectations
        "We need to figure out how to pass mu_F and v_F properly to var_exp"
        "If we follow a similar code for var_exp, then mu_F and v_F have to be lists with"
        "the means and variances of all q_F respectively"
        VE = HeteroLikelihood.var_exp(Y, q_F.mu_F, q_F.v_F)
        for d in range(D):  #Here D number of outputs
            #TODO: Check how to use batch_scale properly when optimising with mini-batches
            VE[d] = VE[d] * batch_scale[d]
            VE_dm[d] = VE_dm[d] * batch_scale[d]
            VE_dv[d] = VE_dv[d] * batch_scale[d]

        # Log Marginal log(p(Y))
        Eqf_g = 0
        for d in range(D):
            Eqf_g += VE[d].sum()

        ELBO = Eqf_g - KL

        return ELBO

class GaussianProcess(nn.Module):
    def __init__(self,X,Y,Z,HeteroLikelihood,kernq_list):
        super().__init__()
        "All this init section has to be created"
        "Should we received the list of X and Y or not?"
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Train_mode = True
        #self.lik_std_noise = torch.nn.Parameter(torch.tensor([1.0])) #torch.tensor([0.07])
        self.Y_metadata = HeteroLikelihood.generate_metadata()
        self.kernq_list = kernq_list
    def forward(self,X, noiseless = True):
        if self.Train_mode:

            # This is the GP prior distribution, p_u.m (mean) and p_u.Kuu (kern covariance)
            # We need to decide how to store the data in p_u, maybe p_u.Kuu is Q x M x M
            # If we use a mean then p_u.m is Q x M, if not simply use zero mean and p_u.m is not necessary
            self.p_u = compute_p_u(self.Z,self.kernq_list)

            f_index = self.Y_metadata['function_index'].flatten()
            for j in range(J):
                Xtask = X[f_index[j]]
                q_fdj = self.calculate_q_f()

            self.q_F.append(q_fdj)

            return self.q_u,self.p_u,self.q_F
        else:
            "This section is for the predicting mode of the model"
            return f_mu, f_Cov


