import numpy as np
from datetime import datetime as date
import matplotlib.pyplot as plt
from matplotlib import colors as mapcol
import itertools



########################################
#              Plot Bandit             #
########################################
def plot_bandit(bandit):
    plt.figure(figsize=(12,7))
    plt.xticks(range(0,bandit.nbr_arms), range(0,bandit.nbr_arms))
    plt.plot(range(0,bandit.nbr_arms), bandit.rewards)
    plt.scatter(range(0,bandit.nbr_arms), bandit.rewards)
    plt.show()

########################################
#           KL Divergences             #
########################################
def klBernoulli(mean_1, mean_2, eps=1e-15):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = np.minimum(np.maximum(mean_1, eps), 1-eps)
    y = np.minimum(np.maximum(mean_2, eps), 1-eps)
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

def klGaussian(mean_1, mean_2, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return ((mean_1-mean_2)**2)/(2*sig2)


########################################
#         Argument selectors           #
########################################

def randamax(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if I is None:
        idxs = np.where(V == np.amax(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amax(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx

def randamin(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if I is None:
        idxs = np.where(V == np.amin(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amin(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx


########################################
#             Experiments              #
########################################

class Experiment:
    def __init__(self, sequential_algorithms, bandit,
                 statistics={'mean':True, 'std':True, 'quantile':False, 'pulls':False},
                 quantile_ticks=[0.1, 0.9, 0.5],
                 kullback=None, complexity=False):
        assert len(sequential_algorithms) > 0
        self.algorithms = sequential_algorithms
        self.bandit = bandit
        self.nbr_algo = len(sequential_algorithms)
        self.algo_idx = 0
        self.statistics = {}
        self.regret = None
        self.path = None
        self.horizon = None
        self.nbr_exp = None
        self.complexity = None
        if complexity:
            if kullback is not None:
                self.complexity = bandit.complexity(kullback)
            else:
                self.complexity = bandit.complexity()
        for algo_idx, algo in enumerate(self.algorithms):
            self.statistics[algo_idx] = {'name': algo.name}
            for s in statistics.items():
                if s[1]:
                    self.statistics[algo_idx][s[0]] = None
        stats = self.statistics[0].keys()
        self.stats = stats
        self.regret_flag = ('mean' in stats) or ('std' in stats) or ('quantile' in stats)
        self.quantile_ticks = quantile_ticks
        self.path_flag = 'pulls' in stats
        # Style
        self.marker = itertools.cycle(('D', '.', '*', 's', '^'))
        self.border_style = itertools.cycle((':', '-.', '--', '-'))
        self.color_edges = itertools.cycle(( (66/255, 135/255, 245/255,1), (227/255, 59/255, 30/255,1),
                                           (59/255, 199/255, 28/255, 1), (10/255, 242/255, 227/255, 1),
                                           (221/255, 250/255, 2/255, 1)))
        self.color_faces = itertools.cycle(( (66/255, 135/255, 245/255,0.1), (227/255, 59/255, 30/255,0.1),
                                           (59/255, 199/255, 28/255, 0.1), (10/255, 242/255, 227/255, 0.1),
                                           (221/255, 250/255, 2/255, 0.1)))
        
    def __call__(self, arm, r, t):
        if self.regret_flag:
            self.regret[t] = self.bandit.regrets[arm]
        if self.path_flag:
            self.path[t] = arm

    def run(self, nbr_exp=500, horizon=50):
        self.nbr_exp = nbr_exp
        self.horizon = horizon
        for algo_idx, algo in enumerate(self.algorithms):
            if self.regret_flag:
                regret = np.zeros((nbr_exp, horizon))
            if self.path_flag:
                path = np.zeros((nbr_exp, horizon), int)
            for i in range(nbr_exp):
                if self.regret_flag:
                    self.regret = regret[i]
                if self.path_flag:
                    self.path = path[i]
                
                algo.fit(horizon, experiment=self)
                
                if self.regret_flag:
                    regret[i] = np.cumsum(self.regret)
                if self.path_flag:
                    path[i] = self.path
            
            for k in self.stats:
                if k == 'mean':
                    self.statistics[algo_idx][k] = np.mean(regret, 0)
                elif k == 'std':
                    self.statistics[algo_idx][k] = np.std(regret, 0)
                elif k == 'pulls':
                    self.statistics[algo_idx][k] = path
                elif k == 'quantile':
                    self.statistics[algo_idx][k] = np.quantile(regret, q=self.quantile_ticks, axis=0)
                    
    def plot(self, save=False, complexity=True, dimensions=[(0,1)], functions=[]):
        ticks = np.arange(self.horizon)
        exp_info = f"Horizon = {self.horizon} - Nbr. of experiments = {self.nbr_exp}"
        if 'mean' in self.stats:
            plt.figure(figsize=(12,6))
            if 'std' in self.stats:
                title = "Mean cumulative regret and standard deviation tube\n"
                title += exp_info
                plt.title(title)
                for algo_stats in self.statistics.values():
                    name = algo_stats['name']
                    mean = algo_stats['mean']
                    std = algo_stats['std']
                    color_line = next(self.color_edges)
                    edge_style = next(self.border_style)
                    plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}", marker=next(self.marker), markevery=100, markersize=9, c=color_line)
                    plt.plot(ticks, np.maximum(0, mean-std), ls=edge_style, c=color_line)
                    plt.plot(ticks, mean+std, ls=edge_style, c=color_line)
                    plt.fill_between(ticks, np.maximum(0, mean-std), mean+std, facecolor=next(self.color_faces))
                
            else:
                title = "Mean cumulative regret\n"
                title += exp_info
                plt.title(title)
                for algo_stats in self.statistics.values():
                        name = algo_stats['name']
                        mean = algo_stats['mean']
                        plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}", marker=next(self.marker), markevery=100, markersize=9, c=next(self.color_edges))
            
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound")
            
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_mean_std_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
            
        elif 'std' in self.stats:
            plt.figure(figsize=(12,6))
            title = "Standard deviation of the regret\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                std = algo_stats['std']
                plt.plot(ticks, std, label = name + f" - R = {std[-1]:.2f}")                
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_std_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
        
        if 'quantile' in self.stats:
            plt.figure(figsize=(12,6))
            title = f"Median cumulative regret and quantile {self.quantile_ticks[:-1]} tube\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                quantile = algo_stats['quantile']
                # By convention, the last quantile is the median
                color_line = next(self.color_edges)
                edge_style = next(self.border_style)
                plt.plot(ticks, quantile[-1], label = name + f" - R = {quantile[-1][-1]:.2f}",  marker=next(self.marker), markevery=100, markersize=9, c=color_line)
                nbr_quantile = len(quantile)
                ptr_1 = 0
                ptr_2 = nbr_quantile-2
                fc = next(self.color_faces)
                while ptr_1 < ptr_2:
                    plt.plot(ticks, quantile[ptr_1], ls=edge_style, c=color_line)
                    plt.plot(ticks, quantile[ptr_2], ls=edge_style, c=color_line)
                    plt.fill_between(ticks, quantile[ptr_1], quantile[ptr_2], facecolor=fc)
                    ptr_1 += 1
                    ptr_2 -= 1
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound")
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_quantile_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
        
        if 'all' in self.stats:
            plt.figure(figsize=(12,6))
            title = f"Mean cumulative regret and quantiles {self.quantile_ticks}\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                quantile = algo_stats['quantile']
                # By convention, the last quantile is the median
                color_line = next(self.color_edges)
                edge_style = next(self.border_style)
                mean = algo_stats['mean']
                plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}", marker=next(self.marker), markevery=100, markersize=9, c=color_line)
                plt.plot(ticks, quantile[-1], c=color_line, ls='-.')
                nbr_quantile = len(quantile)
                ptr_1 = 0
                ptr_2 = nbr_quantile-2
                fc = next(self.color_faces)
                while ptr_1 < ptr_2:
                    plt.plot(ticks, quantile[ptr_1], ls=':', c=color_line)
                    plt.plot(ticks, quantile[ptr_2], ls=':', c=color_line)
                    plt.fill_between(ticks, quantile[ptr_1], quantile[ptr_2], facecolor=fc)
                    ptr_1 += 1
                    ptr_2 -= 1
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound", lw=1.7, c="black")
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n, lw=2.5, c='black')
            plt.xlabel('Time step')
            plt.ylabel('Cumulative regret')
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_quantile_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
            
        if 'pulls' in self.stats:
            # basis = np.eye(self.bandit.nbr_arms, dtype=int)
            basis = np.eye(2, dtype=int)
            colors = list(mapcol.TABLEAU_COLORS.values())
            len_colors = len(colors)
            for dims in dimensions:
                plt.figure(figsize=(10,10))
                title = f"Sampling strategy as a random walk - abs.: arm {dims[0]}, ord.: arm {dims[1]}\n"
                title += exp_info
                plt.title(title)
                color_ctr = 0
                alpha = (self.bandit.nbr_arms+0.1*np.log(float(self.horizon)))/self.nbr_exp
                for algo_stats in self.statistics.values():
                    name = algo_stats['name']
                    paths = algo_stats['pulls']
                    for i, p in enumerate(paths):
                        path = np.zeros((2, self.horizon+1), int)
                        for t, arm in enumerate(p):
                            if arm == dims[0]:
                                path[:,t+1] = path[:,t] + basis[0]
                            elif arm == dims[1]:
                                path[:,t+1] = path[:,t] + basis[1]
                        if i == 0:
                            plt.plot(path[0], path[1], alpha=alpha, color=colors[color_ctr], label=f"{name}")
                        else:
                            plt.plot(path[0], path[1], alpha=alpha, color=colors[color_ctr])
                    color_ctr += 1
                    color_ctr = color_ctr % len_colors
                
                leg = plt.legend()
                for l in leg.get_lines():
                    l.set_alpha(1)
                plt.show()
                if save:
                    plt.savefig(f"./images/rw_dim_{dims[0]}_{dims[1]}_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                                bbox_inches='tight')
                plt.close()
                
                
                
                
########################################
#             Experiments              #
########################################

class Experimentl:
    def __init__(self, sequential_algorithms, bandit,
                 statistics={'mean':True, 'std':True, 'quantile':False, 'pulls':False, 'fairness_mean':True,
                            'fairness_quantile':True},
                 quantile_ticks=[0.1, 0.9, 0.5],
                 qclass = {"mean 0.1":np.array([0,1,2,3]), "mean 0.3":np.array([4,5,6,7]),
                           "mean 0.6":np.array([8,9,10,11]), "mean 0.9":np.array([12,13,14,15]) },
                 kullback=None, complexity=False):
        assert len(sequential_algorithms) > 0
        self.algorithms = sequential_algorithms
        self.bandit = bandit
        self.nbr_algo = len(sequential_algorithms)
        self.algo_idx = 0
        self.statistics = {}
        self.regret = None
        self.path = None
        self.horizon = None
        self.nbr_exp = None
        self.complexity = None
        self.qclass = qclass
        if complexity:
            if kullback is not None:
                self.complexity = bandit.complexity(kullback)
            else:
                self.complexity = bandit.complexity()
        for algo_idx, algo in enumerate(self.algorithms):
            self.statistics[algo_idx] = {'name': algo.name}
            for s in statistics.items():
                if s[1]:
                    self.statistics[algo_idx][s[0]] = None
        stats = self.statistics[0].keys()
        self.stats = stats
        self.regret_flag = ('mean' in stats) or ('std' in stats) or ('quantile' in stats)
        self.quantile_ticks = quantile_ticks
        self.path_flag = 'pulls' in stats
        self.fairness_flag = ('fairness_mean' in stats) or ('fairness_quantile' in stats)
        # Style
        self.marker = itertools.cycle(('D', '.', '*', 's', '^'))
        self.border_style = itertools.cycle((':', '-.', '--', '-'))
        self.color_edges = itertools.cycle(( (66/255, 135/255, 245/255,1), (227/255, 59/255, 30/255,1),
                                           (59/255, 199/255, 28/255, 1), (10/255, 242/255, 227/255, 1),
                                           (221/255, 250/255, 2/255, 1)))
        self.color_faces = itertools.cycle(( (66/255, 135/255, 245/255,0.1), (227/255, 59/255, 30/255,0.1),
                                           (59/255, 199/255, 28/255, 0.1), (10/255, 242/255, 227/255, 0.1),
                                           (221/255, 250/255, 2/255, 0.1)))
        
    def __call__(self, arm, r, t):
        if self.regret_flag:
            self.regret[t] = self.bandit.regrets[arm]
        if self.path_flag:
            self.path[t] = arm

    def run(self, nbr_exp=500, horizon=50):
        self.nbr_exp = nbr_exp
        self.horizon = horizon
        qclass = self.qclass
        for algo_idx, algo in enumerate(self.algorithms):
            if self.regret_flag:
                regret = np.zeros((nbr_exp, horizon))
            if self.fairness_flag:
                fairness = np.zeros((nbr_exp, self.bandit.nbr_arms))
            if self.path_flag:
                path = np.zeros((nbr_exp, horizon), int)
            for i in range(nbr_exp):
                if self.regret_flag:
                    self.regret = regret[i]
                if self.path_flag:
                    self.path = path[i]
                
                algo.fit(horizon, experiment=self)
                
                if self.regret_flag:
                    regret[i] = np.cumsum(self.regret)
                if self.fairness_flag:
                    fairness_tmp = algo.nbr_pulls
                    for elems in self.qclass.keys():
                        fairness_tmp[qclass[elems]] = np.sort(fairness_tmp[qclass[elems]])
                    fairness[i] = fairness_tmp
                if self.path_flag:
                    path[i] = self.path
            
            for k in self.stats:
                if k == 'mean':
                    self.statistics[algo_idx][k] = np.mean(regret, 0)
                elif k == 'std':
                    self.statistics[algo_idx][k] = np.std(regret, 0)
                elif k == 'fairness_mean':
                    self.statistics[algo_idx][k] = np.mean(fairness, 0)
                elif k == 'fairness_quantile':
                    self.statistics[algo_idx][k] = np.std(fairness, 0)
                    # self.statistics[algo_idx][k] = np.quantile(fairness, q=self.quantile_ticks, axis=0)
                elif k == 'pulls':
                    self.statistics[algo_idx][k] = path
                elif k == 'quantile':
                    self.statistics[algo_idx][k] = np.quantile(regret, q=self.quantile_ticks, axis=0)
                    
    def plot(self, save=False, complexity=True, dimensions=[(0,1)], functions=[]):
        ticks = np.arange(self.horizon)
        exp_info = f"Horizon = {self.horizon} - Nbr. of experiments = {self.nbr_exp}"
        if 'mean' in self.stats:
            plt.figure(figsize=(12,6))
            if 'std' in self.stats:
                title = "Mean cumulative regret and standard deviation tube\n"
                title += exp_info
                plt.title(title)
                for algo_stats in self.statistics.values():
                    name = algo_stats['name']
                    mean = algo_stats['mean']
                    std = algo_stats['std']
                    color_line = next(self.color_edges)
                    edge_style = next(self.border_style)
                    plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}", marker=next(self.marker), markevery=100, markersize=9, c=color_line)
                    plt.plot(ticks, np.maximum(0, mean-std), ls=edge_style, c=color_line)
                    plt.plot(ticks, mean+std, ls=edge_style, c=color_line)
                    plt.fill_between(ticks, np.maximum(0, mean-std), mean+std, facecolor=next(self.color_faces))
                
            else:
                title = "Mean cumulative regret\n"
                title += exp_info
                plt.title(title)
                for algo_stats in self.statistics.values():
                        name = algo_stats['name']
                        mean = algo_stats['mean']
                        plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}", marker=next(self.marker), markevery=100, markersize=9, c=next(self.color_edges))
            
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound")
            
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_mean_std_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
            
        elif 'std' in self.stats:
            plt.figure(figsize=(12,6))
            title = "Standard deviation of the regret\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                std = algo_stats['std']
                plt.plot(ticks, std, label = name + f" - R = {std[-1]:.2f}")                
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_std_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
        
        if 'quantile' in self.stats:
            plt.figure(figsize=(12,6))
            title = f"Median cumulative regret and quantile {self.quantile_ticks[:-1]} tube\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                quantile = algo_stats['quantile']
                # By convention, the last quantile is the median
                color_line = next(self.color_edges)
                edge_style = next(self.border_style)
                plt.plot(ticks, quantile[-1], label = name + f" - R = {quantile[-1][-1]:.2f}",  marker=next(self.marker), markevery=100, markersize=9, c=color_line)
                nbr_quantile = len(quantile)
                ptr_1 = 0
                ptr_2 = nbr_quantile-2
                fc = next(self.color_faces)
                while ptr_1 < ptr_2:
                    plt.plot(ticks, quantile[ptr_1], ls=edge_style, c=color_line)
                    plt.plot(ticks, quantile[ptr_2], ls=edge_style, c=color_line)
                    plt.fill_between(ticks, quantile[ptr_1], quantile[ptr_2], facecolor=fc)
                    ptr_1 += 1
                    ptr_2 -= 1
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound")
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_quantile_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
        
        if 'all' in self.stats:
            plt.figure(figsize=(12,6))
            title = f"Mean cumulative regret and quantiles {self.quantile_ticks}\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                quantile = algo_stats['quantile']
                # By convention, the last quantile is the median
                color_line = next(self.color_edges)
                edge_style = next(self.border_style)
                mean = algo_stats['mean']
                plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}", marker=next(self.marker), markevery=100, markersize=9, c=color_line)
                plt.plot(ticks, quantile[-1], c=color_line, ls='-.')
                nbr_quantile = len(quantile)
                ptr_1 = 0
                ptr_2 = nbr_quantile-2
                fc = next(self.color_faces)
                while ptr_1 < ptr_2:
                    plt.plot(ticks, quantile[ptr_1], ls=':', c=color_line)
                    plt.plot(ticks, quantile[ptr_2], ls=':', c=color_line)
                    plt.fill_between(ticks, quantile[ptr_1], quantile[ptr_2], facecolor=fc)
                    ptr_1 += 1
                    ptr_2 -= 1
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound", lw=1.7, c="black")
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n, lw=2.5, c='black')
            plt.xlabel('Time step')
            plt.ylabel('Cumulative regret')
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_quantile_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
            
        if 'pulls' in self.stats:
            # basis = np.eye(self.bandit.nbr_arms, dtype=int)
            basis = np.eye(2, dtype=int)
            colors = list(mapcol.TABLEAU_COLORS.values())
            len_colors = len(colors)
            for dims in dimensions:
                plt.figure(figsize=(10,10))
                title = f"Sampling strategy as a random walk - abs.: arm {dims[0]}, ord.: arm {dims[1]}\n"
                title += exp_info
                plt.title(title)
                color_ctr = 0
                alpha = (self.bandit.nbr_arms+0.1*np.log(float(self.horizon)))/self.nbr_exp
                for algo_stats in self.statistics.values():
                    name = algo_stats['name']
                    paths = algo_stats['pulls']
                    for i, p in enumerate(paths):
                        path = np.zeros((2, self.horizon+1), int)
                        for t, arm in enumerate(p):
                            if arm == dims[0]:
                                path[:,t+1] = path[:,t] + basis[0]
                            elif arm == dims[1]:
                                path[:,t+1] = path[:,t] + basis[1]
                        if i == 0:
                            plt.plot(path[0], path[1], alpha=alpha, color=colors[color_ctr], label=f"{name}")
                        else:
                            plt.plot(path[0], path[1], alpha=alpha, color=colors[color_ctr])
                    color_ctr += 1
                    color_ctr = color_ctr % len_colors
                
                leg = plt.legend()
                for l in leg.get_lines():
                    l.set_alpha(1)
                plt.show()
                if save:
                    plt.savefig(f"./images/rw_dim_{dims[0]}_{dims[1]}_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                                bbox_inches='tight')
                plt.close()
        
        if 'fairness_mean' in self.stats:
            for ctr, c in enumerate(self.qclass.keys()):
                plt.figure(figsize=(47,6))
                nbr_c = len(self.qclass)
                # plt.figure(figsize=(10,10))
                title = f"Sampling repartition within the class of {c}\n"
                title += exp_info
                ii = 1
                for algo_stats in self.statistics.values():
                    name = algo_stats['name']
                    pulls = algo_stats['fairness_mean']
                    pulls_q = algo_stats['fairness_quantile']
                    plt.subplot(1, nbr_c, ii)
                    a = pulls_q[self.qclass[c]]
                    yerr = np.vstack([np.minimum(pulls[self.qclass[c]], a), a])
                    plt.bar([ str(i) for i in range(len(self.qclass[c]))], pulls[self.qclass[c]], yerr=yerr)
                    plt.title(name)
                    ii += 1
                    
                plt.suptitle(title, x=0.28, y=1.)
                # leg = plt.legend()
                plt.show()
                if save:
                    plt.savefig(f"./images/fairness_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                                bbox_inches='tight')
                plt.close()
