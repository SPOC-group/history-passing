import numpy as np
from itertools import product
from scipy.optimize import fmin

from bdcm.rules import DEAD, ALIFE, X_POS, Y_POS


def is_rattling_edge(X,Y):
    return not all_equal(X) and not all_equal(Y)

def is_monochromatic_edge(X,Y):
    return 2 * int(X == Y)

def has_same_period(X,Y):
    return X[0] == Y[0]

def bichrom_rat_deg_(b,c):
    return is_rattling_edge(b,c) and not has_same_period(b,c)

def monochrom_rat_deg_(b,c):
    return is_rattling_edge(b,c) and has_same_period(b,c)

def bichrom_stable_deg_(b,c):
    return not is_rattling_edge(b,c) and not has_same_period(b,c)

def monochrom_stable_deg_(b,c):
    return not is_rattling_edge(b,c) and has_same_period(b,c)

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def sum_(X, Y):
    return ((sum(X) / len(X) * 2) - 1) + ((sum(Y) / len(Y) * 2) - 1)

def rattling_(X, Y):
    return int(not all_equal(X)) + int(not all_equal(Y))


def rattling_cycle(X, Y, p):
    return int(not all_equal(X[p:])) + int(not all_equal(Y[p:]))



MAG_cycle = ('mag_cycle', lambda config: lambda X,Y: sum_(X[config['p']:],Y[config['p']:]) )
MAG_cycle0 = ('mag_cycle0', lambda config: lambda X,Y: sum_(X[config['p']:config['p']+1],Y[config['p']:config['p']+1]) )
MAG_cycle1 = ('mag_cycle1', lambda config: lambda X,Y: sum_(X[config['p']+1:config['p']+2],Y[config['p']+1:config['p']+2]) )

MAG_path0 = ('mag_init', lambda config: lambda X,Y: sum_(X[0:1],Y[0:1]) )
MAG_pathend = ('MAG_pathend', lambda config: lambda X,Y: sum_(X[config['p']-1:config['p']],Y[config['p']-1:config['p']]) )
RAT_cycle = ('rat_cycle', lambda config: lambda X,Y: rattling_(X[config['p']:],Y[config['p']:]) )
# mono/bi-chromatic edges
bichrom_rat_deg = ('bichrom_rat_deg', lambda config: lambda X,Y: bichrom_rat_deg_(X[config['p']:],Y[config['p']:]) )
monochrom_rat_deg =  ('monochrom_rat_deg', lambda config: lambda X,Y: monochrom_rat_deg_(X[config['p']:],Y[config['p']:]) )
bichrom_stable_deg =  ('bichrom_stable_deg', lambda config: lambda X,Y: bichrom_stable_deg_(X[config['p']:],Y[config['p']:]) )
monochrom_stable_deg =  ('monochrom_stable_deg', lambda config: lambda X,Y: monochrom_stable_deg_(X[config['p']:],Y[config['p']:]) )
MONO_cycle1 = ('monoch_edges_cycle1', lambda config: lambda X,Y: is_monochromatic_edge(X[config['p']:config['p']+1],Y[config['p']:config['p']+1]) )
MONO_cycle2 = ('monoch_edges_cycle2', lambda config: lambda X,Y: is_monochromatic_edge(X[config['p']+1:config['p']+2],Y[config['p']+1:config['p']+2]) )
MONO_pathend = ('monoch_edges_pathend', lambda config: lambda X,Y: is_monochromatic_edge(X[config['p']-1:config['p']],Y[config['p']-1:config['p']]) )
rattler_deg =  ('rattler_deg', lambda config: lambda center,neighbours: bichrom_stable_deg_(X[config['p']:],Y[config['p']:]) )
MAG_RSB_comp = ("rho_0", lambda config: lambda s_i, s_j: (s_i[0] + s_j[0]))


class Observables:

    cycle_obs = [MAG_cycle,MAG_cycle0,RAT_cycle, bichrom_rat_deg, monochrom_rat_deg, bichrom_stable_deg, monochrom_stable_deg, MONO_cycle1]
    default_obs = [MAG_path0, MAG_RSB_comp]
    path_obs = [ MAG_pathend,MONO_pathend]
    def __init__(self, **config):

        self.obs = [] + self.default_obs
        if config['p'] > 0:
            self.obs += self.path_obs
        if config['c'] != 0:
            self.obs += self.cycle_obs
        if config['c'] == 2:
            self.obs += [MONO_cycle2,MAG_cycle1]

        self.obs_map = {v[0]: i for i, v in enumerate(self.obs)}
        self.names = [o[0] for o in self.obs]
        self.temps = np.array([config.get(name+'_temp',0.0) for name in self.names])
        self.funcs = [o[1](config) for o in self.obs]
        self.targets = np.array([config.get(name+'_target',0.0) for name in self.names])

        self.attr_size_graph = config['attr_size_graph']
        self.d = config['d']

    def measure(self,b,c):
        return np.array([f(b,c) for f in self.funcs])
    def g(self, X, Y):
        #print(X,Y)
        #print(self.measure(X, Y))
        return np.exp(-self.temps / self.d * self.measure(X, Y))

    def Z_ij_prime_(self, chi):
        Z = np.zeros_like(self.temps)
        for X in product([DEAD, ALIFE], repeat=self.attr_size_graph):
            for Y in product([DEAD, ALIFE], repeat=self.attr_size_graph):
                idx_a = [0] * (2 * self.attr_size_graph)
                idx_b = [0] * (2 * self.attr_size_graph)

                # assignment stays the same, but direction is different: x->y, y->x
                idx_a[X_POS::2] = X
                idx_a[Y_POS::2] = Y
                idx_b[X_POS::2] = Y
                idx_b[Y_POS::2] = X

                Z += (chi[tuple(idx_a)] * chi[tuple(idx_b)]) * (1.0 / self.g(X, Y)).prod() * self.measure(Y,X)

        return Z

    def Z_ij_prime_pop(self, chi1,chi2):
        Z = np.zeros_like(self.temps)
        for X in product([DEAD, ALIFE], repeat=self.attr_size_graph):
            for Y in product([DEAD, ALIFE], repeat=self.attr_size_graph):
                idx_a = [0] * (2 * self.attr_size_graph)
                idx_b = [0] * (2 * self.attr_size_graph)

                # assignment stays the same, but direction is different: x->y, y->x
                idx_a[X_POS::2] = X
                idx_a[Y_POS::2] = Y
                idx_b[X_POS::2] = Y
                idx_b[Y_POS::2] = X

                Z += (chi1[tuple(idx_a)] * chi2[tuple(idx_b)]) * (1.0 / self.g(X, Y)).prod() * self.measure(Y, X)
        return Z

    def Z_ij_(self, chi):
        Z = 0
        for X in product([DEAD, ALIFE], repeat=self.attr_size_graph):
            for Y in product([DEAD, ALIFE], repeat=self.attr_size_graph):
                idx_a = [0] * (2 * self.attr_size_graph)
                idx_b = [0] * (2 * self.attr_size_graph)

                # assignment stays the same, but direction is different: x->y, y->x
                idx_a[X_POS::2] = X
                idx_a[Y_POS::2] = Y
                idx_b[X_POS::2] = Y
                idx_b[Y_POS::2] = X

                Z += (chi[tuple(idx_a)] * chi[tuple(idx_b)]) * (1.0 / self.g(X, Y)).prod()

        return Z

    def Z_ij_pop(self, chi1,chi2):
        Z = 0
        for X in product([DEAD, ALIFE], repeat=self.attr_size_graph):
            for Y in product([DEAD, ALIFE], repeat=self.attr_size_graph):
                idx_a = [0] * (2 * self.attr_size_graph)
                idx_b = [0] * (2 * self.attr_size_graph)

                # assignment stays the same, but direction is different: x->y, y->x
                idx_a[X_POS::2] = X
                idx_a[Y_POS::2] = Y
                idx_b[X_POS::2] = Y
                idx_b[Y_POS::2] = X

                Z += (chi1[tuple(idx_a)] * chi2[tuple(idx_b)]) * (1.0 / self.g(X, Y)).prod()

        return Z

    def calc_marginals(self,chi):
        Z_ij_prime = self.Z_ij_prime_(chi)
        Z_ij = self.Z_ij_(chi)

        observables = 0.5 * Z_ij_prime / Z_ij
        return {
            **{name: o for name, o in zip(self.names, observables)},
            **{name + '_Z_ij_prime': t for name, t in zip(self.names,observables)},
            'Z_ij': Z_ij,
            'Legendre': (observables * self.temps).sum() # Legendre transform
        }

    def calc_marginals_pop(self,chi1,chi2):
        Z_ij_prime = self.Z_ij_prime_pop(chi1,chi2)
        Z_ij = self.Z_ij_pop(chi1,chi2)

        observables = 0.5 * Z_ij_prime / Z_ij
        return {
            **{name: o for name, o in zip(self.names, observables)},
            **{name + '_Z_ij_prime': t for name, t in zip(self.names,observables)},
            'Z_ij': Z_ij,
            'Legendre': (observables * self.temps).sum() # Legendre transform
        }

    def update_best_temps(self,observable, observable_target, chi):
        # optimize for the different functions
        idx = self.obs_map[observable]

        def f(temp):
            self.temps[idx] = temp
            Z_ij_prime = self.Z_ij_prime_(chi)
            Z_ij = self.Z_ij_(chi)
            observables = 0.5 * Z_ij_prime / Z_ij
            return (observable_target - observables[idx]) ** 2

        optimal_temp = fmin(f, 0.0, disp=False, ftol=0.00001, xtol=0.00001)[0]
        self.temps[idx] = -1 * 2 * optimal_temp # dont ask why the -1 is here. It does not work without it.
