import numpy as np
from forban.sequentialg import SequentiAlg
from forban.utils import *
import itertools
from scipy.optimize import linprog

########################################
#                UCB                   #
########################################

class UCB(SequentiAlg):
    def __init__(self, bandit, name="UCB",
                 params={'init': np.inf, 'sigma':1, 'inv_delta': lambda t: (t+1)**2,
                     'alpha':2 }):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.sigma = params['sigma']
        self.inv_delta = params['inv_delta']
        self.alpha = params['alpha']

    def compute_indices(self):
        if self.all_selected:
            # Exploration bonuses
            bonus = np.sqrt(self.alpha*np.log(self.inv_delta(self.time))/self.nbr_pulls)
            bonus *= self.sigma
            # Indexes
            self.indices = self.means + bonus
        else:
            for arm in np.where(self.nbr_pulls != 0)[0]:
                bonus = np.sqrt(self.alpha*np.log(self.inv_delta(self.time))/self.nbr_pulls[arm])
                self.indices[arm] = self.means[arm] + bonus

    def choose_an_arm(self):
        return randamax(self.indices)



class IMED(SequentiAlg):
    def __init__(self, bandit, name="IMED", params={'init': -np.inf, 'kl':klGaussian}):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.kl = params['kl']

    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            self.indices = self.nbr_pulls*self.kl(self.means, max_mean) + np.log(self.nbr_pulls)
        else:
            for arm in np.where(self.nbr_pulls != 0)[0]:
                self.indices[arm] = self.nbr_pulls[arm]*self.kl(self.means[arm], max_mean) \
                + np.log(self.nbr_pulls[arm])


class CombinatorialIMED(SequentiAlg):
    def __init__(self, bandit, name="Comb. IMED",
            params={'init': -np.inf, 'kl':klGaussian, 'c':3}):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.kl = params['kl']
        self.c = params['c']
        self.nepc = bandit.nbr_arms // self.c # nbr. of elements per class
        self.cl = [np.arange(self.nepc*i, self.nepc*(i+1)) for i in range(self.c)]
        self.class_indices = np.zeros(self.c)
        self.arms = np.arange(bandit.nbr_arms)
    
    def compute_indices(self):
        self.indices = self.nbr_pulls
    
    def choose_an_arm(self):
        if self.all_selected:
            max_mean = np.max(self.means)
            a_star = randamax(self.means)
            I_star = np.log(self.nbr_pulls[a_star])
            
            I_bar = I_star
            kl_idx = self.kl(self.means, max_mean)
            for elms in itertools.combinations(self.arms, self.nepc):
                elms = list(elms)
                n = self.nbr_pulls[elms]
                kl_idx_slice = kl_idx[elms]
                N = n.sum()
                combinatoric_imed_idx = (n*iidx).sum() + np.log(N) 
                I_bar = min(combinatoric_imed_idx, I_bar)
            
            if I_star <= I_bar:
                return a_star
            else:
                x = self.nbr_pulls*kl_idx + np.log(self.nbr_pulls)
                x[permutation[-1]] = np.inf
                return randamin(x)
        else:
            return randamin(self.indices)


class RelaxedCombinatorialIMED(SequentiAlg):
    def __init__(self, bandit, name="IMED-EC",
            params={'init': -np.inf, 'kl':klGaussian, 'c':3, 'nepc':2}):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.kl = params['kl']
        # self.c = params['c']
        self.nepc = params['nepc'] # bandit.nbr_arms // self.c # nbr. of elements per class
        # self.cl = [np.arange(self.nepc*i, self.nepc*(i+1)) for i in range(self.c)]
        # self.class_indices = np.zeros(self.c)
        self.arms = np.arange(bandit.nbr_arms)
    
    def compute_indices(self):
        self.indices = self.nbr_pulls
    
    def choose_an_arm(self):
        if self.all_selected:
            max_mean = np.max(self.means)
            a_star = randamax(self.means, T=self.nbr_pulls)
            I_star = np.log(self.nbr_pulls[a_star])

            imed_idx = self.nbr_pulls*self.kl(self.means, max_mean) + self.nbr_pulls
            nepc_smallest = np.argpartition(imed_idx, self.nepc)[:self.nepc]
            I_bar = imed_idx[nepc_smallest].sum()
            
            if I_star <= I_bar:
                return a_star
            else:
                imed_idx[np.argwhere(self.means == max_mean).flatten()] = np.inf
                return randamin(imed_idx)
        else:
            return randamin(self.indices)
        
#### KLUCB ####       
def klucb(x, level, div, upperbound, lowerbound=-float('inf'), precision=1e-6):
    """Generic klUCB index computation using binary search: 
    returns u>x such that div(x,u)=level where div is the KL divergence to be used.
    """
    l = max(x, lowerbound)
    u = upperbound
    while u-l>precision:
        m = (l+u)/2
        if div(x, m)>level:
            u = m
        else:
            l = m
    return (l+u)/2


def klucbBernoulli(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Bernoulli kl-divergence."""
    upperbound = min(1.,x+sqrt(level/2)) 
    return klucb(x, level, klBernoulli, upperbound, precision)


def klucbGaussian(x, level, sig2=1., precision=0.):
    """returns u such that kl(x,u)=level for the Gaussian kl-divergence (can be done in closed form).    
    """
    return x + np.sqrt(2*sig2*level)

########################################
#               KL-UCB                 #
########################################

class KLUCB(SequentiAlg):
    def __init__(self, bandit, name="KL-UCB",
                 params={'init': np.inf, 'klsolver':klucbGaussian }):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.kl_solver = params['klsolver']

    def compute_indices(self):
        if self.all_selected:
            for arm in range(self.bandit.nbr_arms):
                level = np.log(self.time) / self.nbr_pulls[arm]
                self.indices[arm] = self.kl_solver(self.means[arm], level)
        else:
            for arm in np.where(self.nbr_pulls != 0)[0]:
                level = np.log(self.time) / self.nbr_pulls[arm]
                self.indices[arm] = self.kl_solver(self.means[arm], level)

    def choose_an_arm(self):
        return randamax(self.indices)
    
    
### OSSB
########################################
#                 OSSB                 #
########################################

def klGaussian(mean_1, mean_2, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return ((mean_1-mean_2)**2)/(2*sig2)

class OSSB(SequentiAlg):
    def __init__(self, bandit, name="OSSB",
                 params={'init': np.inf, 'nepc':4, 'kl':klGaussian, 'gamma':0.1, 'epsilon': 0.001 }):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.nepc = params['nepc']
        self.arms = np.arange(bandit.nbr_arms)
        self.s = 0
        self.kl = params['kl']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

    def compute_indices(self):
        if self.all_selected:
            max_mean = np.max(self.means)
            delta = max_mean - self.means
            c = np.zeros(self.bandit.nbr_arms)
        
            for elms in itertools.combinations(self.arms, self.nepc):
                # print(elms)
                kl_tmp = np.zeros(self.bandit.nbr_arms)
                elms = np.array(list(elms)) # elems are the nepc arms moving up
                kl_tmp[elms] = self.kl(self.means[elms], max_mean)
                
                el = 0
                ags = np.argsort(self.means)
                means_tmp = self.means[ags]
                elms_ags = ags[elms]
                for _ in range(self.bandit.nbr_arms // self.nepc - 1):
                    ctr = 0
                    idx = []
                    while ctr < self.nepc:
                        if el not in elms_ags:
                            ctr += 1
                            idx.append(el)
                            el += 1
                        else:
                            el += 1
                    # print(idx, elms_ags)
                    idx = np.array(idx)
                    m = means_tmp[idx]
                    x = np.mean(m)
                    kl_tmp[ags[idx]] = self.kl(m,x)
                    # print(kl_tmp)
                    
                res = linprog(delta.reshape(-1, 1), A_ub=-kl_tmp.reshape(1, -1), b_ub=-1, bounds=(0, None))
                c = np.maximum(c, res.x.reshape(-1))
            # print("res", c)
            self.indices = self.nbr_pulls / c
        else:
            self.indices = self.nbr_pulls

    def choose_an_arm(self):
        if self.all_selected:
            if np.all(self.indices >= (1 + self.gamma)*np.log(self.time)):
                return randamax(self.means)
            else:
                self.s += 1
                ub = randamin(self.indices)
                lb = randamin(self.nbr_pulls)
                if self.nbr_pulls[lb] <= self.epsilon * self.s:
                    return lb
                else:
                    return ub
        else:
            return randamin(self.nbr_pulls)
