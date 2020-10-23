import numpy as np
from functools import partial
import random


class BO:
    ''' Wrapper over various bayesian optimization methods.'''
    
    def __init__(self, kernel='rbf', m=0, lam=1.0, rho=1.0):
        
        self.kernel = kernel
        self.inf_kernel = False
        self.lam = lam
        self.t = 0
        self.rho = rho

        self.m = m
        self.feat_func = lambda x: x
        
        self.reset()

    def set_feat_func(self, func):
        if not self.inf_kernel:
            self.feat_func = func

    def reset(self):
        ''' Reset the BO parameters '''
        self.t = 0
        self.S_t = self.lam * np.eye(self.m)
        self.u_t = np.zeros((1, self.m))
        self.X_t = None
    
    def _update_fin_dim(self, x_t, y_t, training=False):
        # regular finite-dimensional update

        if not training:
            self.t += 1

        phi_t = self.feat_func(x_t)
        y_t = np.expand_dims(y_t, 0)

        if self.X_t is None:
            self.X_t = phi_t
            self.y_t = y_t
        else:
            self.X_t = np.concatenate((self.X_t, phi_t), axis=0)
            self.y_t = np.concatenate((self.y_t, y_t), axis=0)
        
        self.S_t += np.matmul(phi_t, phi_t.T)
        self.u_t += y_t * phi_t
    
    def update(self, x_t, y_t, training=False):
        ''' Update internal parameters with new observations.'''
        self._update_fin_dim(x_t, y_t, training)
    
    def _params_fin(self):
        # return mu, sigma for finite k
        S_inv = np.linalg.inv(self.S_t)
        mu = np.matmul(S_inv, np.matmul(self.X_t.T, self.y_t))

        return mu, S_inv
    
    def get_posterior(self, D_t):
        ''' Compute the posterior mean and variance for each arm in D_t '''
        
        mus, sigmas = [], []
        if not self.inf_kernel:
            u, S_inv = self._params_fin()

        for x in D_t:
            
            phi_t = self.feat_func(x)
            mu = np.dot(phi_t, u)
            sigma = np.matmul(phi_t, np.matmul(S_inv, phi_t.T))
            
            mus.append(mu)
            sigmas.append(sigma)

        return mus, sigmas


class NoisyBO(BO):
    ''' Private Bayesian Optimization (BO) via posterior noise'''

    def get_posterior(self, D_t):
        ''' Compute the noisy posterior mean and variance for each arm in D_t '''
        
        mus, sigmas = [], []
        u, S_inv = self.u_t, np.linalg.inv(self.S_t)
        for x in D_t:
            
            phi_t = self.feat_func(x)
            mu = np.dot(phi_t, u)
            sigma = np.matmul(phi_t, np.matmul(S_inv, phi_t.T))
            
            mus.append(mu)
            sigmas.append(sigma)

        return mus, sigmas


class Agent:

    def __init__(self):
        
        self.t = 0
        pass

    def select_action(self, *args, **kwargs):
        self.t += 1
    
    def update(self, *args, **kwargs):
        pass

    def reset(self):
        pass


class Random(Agent):

    def __init__(self):
        super(Random, self).__init__()
    
    def select_action(self, D_t, *args, **kwargs):
        # randomly select action
        super(Random, self).select_action()
        return random.choice(D_t)


class FedUCB(Agent):
    ''' One agent of FedUCB with JDP.'''

    def __init__(
            self, T=1000, lam=1.0, m=0, B=1.0, epsilon=0.01, beta=0.01, 
            rho=1.0, beta_mult=0.01):

        super(FedUCB, self).__init__()
        self.bo = NoisyBO('linear', m, lam, rho=rho)
        self.delta = 0.01
        self.B = B
        self.T = T

        self.epsilon = epsilon
        self.beta = beta
        self.beta_mult = beta_mult

        self.dp_std = np.sqrt(16*(1+np.ceil(np.log(self.T)))
            *(2+(self.B**2)+2*(self.bo.rho**2))
            *(np.log(10.0/self.beta)**2)/(self.epsilon**2))
        
        self.dim_noise = self.bo.m + 1
        self.N_t = np.ones((self.dim_noise, self.dim_noise))\
            * 2*self.dp_std*(np.sqrt(np.log(self.T) * 2)\
            * (4*np.sqrt(self.bo.m)
            + 2*np.log(2*self.T/self.delta)))
        self.num_samples = 0
        self.num_sync_rounds = 0

    
    def update(self, x_t, y_t, training):
        self.bo.update(x_t, y_t, training=training)
    
    def reset(self):
        self.bo.reset()
        self.N_t = np.ones((self.dim_noise, self.dim_noise))\
            * 2*self.dp_std*(np.sqrt(np.log(self.T) * 2)\
            * (4*np.sqrt(self.bo.m)
            + 2*np.log(2*self.T/self.delta)))
        self.num_samples = 0

    def sample_noise(self):
        
        self.num_sync_rounds += 1
        n_samples_needed = 1 + np.ceil(np.log2(self.num_sync_rounds))

        while self.num_samples < n_samples_needed:
            self.num_samples += 1
            N_t = np.random.normal(
            loc=0, scale=self.dp_std, size=(self.dim_noise, self.dim_noise))
            N_t = (N_t + N_t.T)/np.sqrt(2)
            self.N_t += N_t


    def get_beta(self):

        return self.bo.rho * \
                np.sqrt(self.bo.m * np.log(1 + self.bo.t * (self.B**2) / self.bo.lam) 
                - np.log(self.delta)) + self.bo.lam ** (0.5) * self.B
    
    def select_action(self, D_t, **kwargs):
        
        if self.bo.t == 0:
            return random.choice(D_t)

        mus, sigmas = self.bo.get_posterior(D_t)
        
        beta_t = self.get_beta() * self.beta_mult
        best_x, max_ucb = None, -np.inf
        for x, mu, sigma in zip(D_t, mus, sigmas):
            ucb_x  = mu + beta_t * sigma
            if max_ucb < ucb_x:
                best_x, max_ucb = x, ucb_x

        return best_x
    
    def send_params(self):
        ''' Send noisy parameters to server. '''
        self.sample_noise()
        
        H_t = self.N_t[:self.bo.m, :self.bo.m]
        h_t = np.expand_dims(self.N_t[:self.bo.m, -1], axis=1)
        
        return self.bo.u_t + h_t, self.bo.S_t + H_t
    
    def get_params(self, u_t, S_t):
        ''' Get parameters from server. '''
        self.bo.u_t = u_t
        self.bo.S_t = S_t