import numpy as np
import random

from .helpers import normalize

class Env:

    def __init__(self, n=5, d=5, B=1.0, noise='normal'):
        
        self.n = n
        self.d = d
        self.B = B

        # initialize kernel parameters
        self.init_kernel()

        # initialize noise parameters
        self.noise = noise
        self.init_noise()

        self.actions = []

        # initialize monitor
        self.regrets = []

    
    def init_kernel(self):
        ''' initialize the index set for kernel'''
        _x = np.random.random_sample((1, self.d))
        _norms = np.linalg.norm(_x, ord=2, axis = 1, keepdims = True)
        self.theta = _x * self.B / _norms

    def init_noise(self):

        if self.noise == 'normal':
            self.rho = 1.0
    
    def f(self, x):
        ''' get dot product with f'''        
        return np.dot(self.theta, x)
    
    def sample_action(self):
        x_i = np.random.random_sample((1, self.d))
        # normalize to l2 ball
        x_i = normalize(x_i, p=2)
        return x_i

    def get_action_set(self):

        actions = []
        opt_x = None
        subopt_x = []
        c_opt = 0

        while opt_x is None or len(subopt_x) < self.n - 1:
            x_i = np.random.random_sample((1, self.d))
            # normalize to l2 ball
            x_i = normalize(x_i, p=2)
            c_opt += 1
            if self.f(x_i) >= 0.8:
                opt_x = x_i
            if self.f(x_i) <= 0.7:
                subopt_x.append(x_i)
            if c_opt > 500:
                print('Restarting kernel init')
                c_opt = 0
                self.init_kernel()
        print('Tries for optimal arm: %d' % c_opt)

        
        actions.append(opt_x)
        actions.extend(subopt_x)
        random.shuffle(actions)
        
        # calculate best action and store latest r*
        round_rewards = [self.f(x) for x in actions]
        self.opt_x = np.argmax(round_rewards)
        self.opt_r = round_rewards[self.opt_x]
        self.actions = actions

        return actions
    
    def sample_noise(self, f_x):

        if self.noise == 'normal':
            return f_x + np.random.normal(scale=self.rho)
        
        if self.noise == 'bernoulli':
            return np.random.binomial(1, f_x) 

    
    def play(self, x_t):
        f_x = self.f(x_t)

        y_t = self.sample_noise(f_x)
        r_t = self.opt_r - f_x

        return y_t, r_t

class GridEnv(Env):

    def get_action_set(self):
        pass