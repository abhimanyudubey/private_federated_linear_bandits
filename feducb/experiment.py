import private_gp as gp
import numpy as np

import matplotlib
from .helpers import matplotlib_init
import matplotlib.pyplot as plt


class Experiment:
    
    def __init__(self, exp_name, T, n=10, d=10, m=0, k=5, kernel='rbf', noise='normal', training=0):
        
        self.T = T
        self.n = n
        self.d = d
        self.m = m
        self.k = k
        self.kernel = kernel
        self.noise = noise
        
        self.env = gp.Env(n, d, kernel=self.kernel, noise=self.noise)
        self.exp_name = exp_name
        
        self.algs = []
        self.alg_names = []

        self.training = training
    
    def add_algorithm(self, alg, alg_name):
        ''' Add new algorithm to experiment'''
        self.algs.append(alg)
        self.alg_names.append(alg_name)
    
    def run(self, return_complete=False):
        ''' Runs the GP environment for k experiments and T trials'''
            
        regrets = None
        for kk in range(self.k):
            
            self.env = gp.Env(
                self.n, self.d, kernel=self.kernel, noise=self.noise)
            
            D_t = self.env.get_action_set()
            
            reg_roundx = [[] for _ in range(len(self.algs))]
            for alg in self.algs:
                alg.reset()
            
            print('Running round %d of %d' % (kk+1, self.k))

            for i in range(self.training):
                x_t = self.env.sample_action()
                y_t, _ = self.env.play(x_t)

                for alg in self.algs:
                    alg.update(x_t, y_t, training=True)
            # training completed
            print('Training done.')
            
            for t in range(self.T):

                if t % 1000 == 0:
                    print('Iteration %d of %d' % (t+1, self.T))
            
                for i, alg in enumerate(self.algs):

                    x_t = alg.select_action(D_t, beta_mult=0.1)
                    y_t, r_t = self.env.play(x_t)
                    
                    alg.update(x_t, y_t, training=False)

                    if len(reg_roundx[i]) > 0:
                        reg_roundx[i].append(reg_roundx[i][-1] + r_t)
                    else:
                        reg_roundx[i].append(r_t)
            
            regrets_trial = np.array(reg_roundx)
            regrets_trial = np.expand_dims(regrets_trial, axis=2)
            
            if regrets is None:
                regrets = regrets_trial
            else:
                regrets = np.concatenate((regrets, regrets_trial), axis=2)
        
        # calculate mean and standard deviation across rounds
        means = np.mean(regrets, axis=2)
        stdevs = np.std(regrets, axis=2)
        
        if return_complete:
            return means, stdevs, regrets

        self.means = np.squeeze(means)
        self.sigmas = np.squeeze(stdevs)
        
    def plot_figure(self):
        ''' Plot figure in matplotlib '''
        matplotlib_init()
        
        if not hasattr(self, 'means'):
            print('Please run experiment first')
            return 1
        
        for i, alg_name in enumerate(self.alg_names):
            
            plt.plot(range(self.T), self.means[i], label=alg_name)
            
            conf_min = self.means[i] - 0.5 * self.sigmas[i]
            conf_max = self.means[i] + 0.5 * self.sigmas[i]
            
            plt.fill_between(range(self.T), y1=conf_min, y2=conf_max, alpha=0.15, 
                linewidth=0)
        
        plt.legend(loc='upper left')
        plt.title(self.exp_name)