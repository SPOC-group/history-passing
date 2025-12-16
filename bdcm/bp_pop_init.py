import numpy as np
from tqdm import tqdm
from itertools import product
import math
from dataclasses import dataclass
import networkx as nx
from collections.abc import Callable
from bdcm.dynamics import evolve_dynamics
import time
import wandb
import pandas as pd
import matplotlib.pyplot as plt

BRIBED = 1
HONEST = 0

ZERO = 0
ONE = 1


def bribe_rule(args, s_i_t, s_i_tp1, s_j_t):

    d = args.d
    if s_j_t is not None:
        # return the ranges of neighbours that are ok
        if s_i_tp1 == ONE:
            # everything that is larger than half is fine, less if s_j is already one, less if s_i_t is already one (always stay)
            if s_i_t == ONE:
                return slice(math.ceil(d / 2) - s_j_t , None)
            elif s_i_t == ZERO:
                return slice(math.ceil(d / 2 + 0.5) - s_j_t, None)
        elif s_i_tp1 == ZERO:
            if s_i_t == ONE:
                return slice(None, math.floor(d / 2  + 0.5) - s_j_t  + 1)
            elif s_i_t == ZERO:
                return slice(
                    None, math.floor(d / 2) - s_j_t + 1
                ) # excluding, check here!
    else:
        if s_i_tp1 == ONE:
            if s_i_t == ONE:
                return slice(math.ceil(d / 2) , None)
            elif s_i_t == ZERO:
                return slice(math.ceil(d / 2 + 0.5), None)
        if s_i_tp1 == ZERO:
            if s_i_t == ONE:
                return slice(None, math.floor(d / 2  - 0.5)  + 1)
            elif s_i_t == ZERO:
                return slice(
                    None, math.floor(d / 2) + 1
                ) #
    raise ValueError("should not happen")


def majority_rule(args, x_i, x_j, s_i_t, s_i_tp1, s_j_t):


    d = args.d

    if s_j_t is not None:
        # return the ranges of neighbours that are ok
        if s_i_tp1 == ONE:
            # everything that is larger than half is fine, less if s_j is already one, less if s_i_t is already one (always stay)
            return slice(math.ceil(d / 2) - s_j_t, None)
        if s_i_tp1 == ZERO:
            return slice(None, math.floor(d / 2) - s_j_t , None)
    else:
        if s_i_tp1 == ONE:
            # everything that is larger or equal to half is fine, less if s_j is already zero
            return slice(math.ceil(d / 2) , None)
        if s_i_tp1 == ZERO:
            return slice(None, math.floor(d / 2) + 1)  # excluding, check here!

    raise ValueError("should not happen")


def closing_condition(s_i_t, s_i_tp1, s_j_t, args):
    if args.loops is None:
        return slice(None)
    if args.loops == "homogenous":
        return args.rule(args, s_i_t, ONE, s_j_t)
    if args.loops == "c-1":
        return args.rule(args, s_i_t, s_i_tp1, s_j_t)


def compute_phi(phi, phis_local, args,return_Z=False):

    # N: batch dimension
    # phi: N x 2 x 2 x 2^T x 2^T
    # phis_local: N x d-1 x 2 x 2 x 2^T x 2^T
    # mask: which messages are illegal, to be used with phi

    rule = args.rule
    T = args.T
    d = args.d

    phi_new = np.zeros_like(phi)

    # first construct the array where we get P(x_i,s_i,sum_1, sum_2,sum_3,..)
    # the sums are how many s's are 1 at time step T, we compute it using dynamical programming

    # probability_alife
    # first part is the probabilty of the center node, the rest is the aggregated neighbours
    # x_i, s_i, sum_{neighbours}
    # shape: 2 x 2^T x k^1 x ... x k^T
    # x_i, s_i^1, ... s_i^T, k^1, ... k^T  # where k is the sum of the neighbours which can grow up to d-1
    prob_alife = np.zeros([phi.shape[0]] + [2] * T + [d] * T)
    
    for s_k in product([ZERO, ONE], repeat=T):
        for s_i in product([ZERO, ONE], repeat=T):
            idx_k_to_i = tuple(
                [slice(None), 0] + list(s_k) + list(s_i)
            )
            idx_prob = tuple(
                [slice(None)] + list(s_i) + [s for s in s_k]
            )  # int(s or x_k) # the or is the bribe!

            prob_alife[idx_prob] += phis_local[idx_k_to_i]
    
    
    for k in range(1, d - 1):
        prob_alife__ = np.zeros_like(prob_alife)
        for s_k in product([ZERO, ONE], repeat=T):
            for s_i in product([ZERO, ONE], repeat=T):
                idx_k_to_i = tuple(
                    [slice(None), k] + list(s_k) + list(s_i)
                )

                idx_kp1 = tuple(
                    [slice(None)]
                    + list(s_i)
                    + [slice(s_k[p], k + 1 + s_k[p]) for p in range(T)]
                )  # int( s_k[p] or x_k) the or is the bribe!
                idx_k = tuple(
                    [slice(None)]
                    + list(s_i)
                    + [slice(None, k + 1) for _ in range(T)]
                )

                prob_alife__[idx_kp1] += (
                    prob_alife[idx_k]
                    * phis_local[idx_k_to_i][
                        tuple([slice(None)] + [np.newaxis] * T)
                    ]
                )  
        prob_alife = prob_alife__
    
    # only use the probabilities that fit the rule!
    phi_new = np.zeros_like(phi)
    for s_j in product([ZERO, ONE], repeat=T):
        for s_i in product([ZERO, ONE], repeat=T):
            idx_i_to_j = tuple([slice(None)] + list(s_i) + list(s_j))
            idx_k = tuple(
                [slice(None)]
                + list(s_i)
                + [
                    rule(args, s_i_t, s_i_tp1, s_j_t)
                    for s_i_t, s_i_tp1, s_j_t in zip(
                        list(s_i)[:-1], list(s_i)[1:], list(s_j[:-1])
                    )
                ]
                + [closing_condition(s_i[-1], s_i[-1], s_j[-1], args)]
            )
            
            phi_new[idx_i_to_j] += prob_alife[idx_k].sum(
                axis=tuple(range(1, T + 1))
            )
    
    for s_j in product([ZERO, ONE], repeat=T):
        for s_i in product([ZERO, ONE], repeat=T):
            idx_i_to_j = tuple([slice(None)] + list(s_i) + list(s_j))

            phi_new[idx_i_to_j] *= args.observables.measure(
                s_i, s_j
            ).prod()
    
    Z = np.sum(phi_new, axis=tuple(range(1, phi_new.ndim)), keepdims=True)
    
    phi_new = normalize(phi_new)
    
    if return_Z:
        return phi_new, Z
    
    return phi_new

def compute_marginals(phi_ijs, neighbourhoods, args):
    # phi ijs is the messages from i to j and j to i
    # d*n x 2
    # neighbourhood is for each node, the neighbourhood edge index it belongs to
    # n x d
    T = args.T
    
    Z_ijs = np.zeros((phi_ijs.shape[0], 2))
    for s_j in product([ZERO, ONE], repeat=T):
        for s_i in product([ZERO, ONE], repeat=T):
            idx_i_to_j = tuple(
                [slice(None), 0] + list(s_i) + list(s_j)
            )
            idx_j_to_i = tuple(
                [slice(None), 1] + list(s_j) + list(s_i)  # order ... ??
            )
            Z_ijs[:,s_i[0]] += (
                phi_ijs[idx_i_to_j]
                * phi_ijs[idx_j_to_i]
                #* 1 / args.observables.measure(x_i, x_j, s_i, s_j).prod()
            )
    Z_ijs /= Z_ijs.sum(axis=-1,keepdims=True)
    
    marginals = np.zeros((neighbourhoods.shape[0], 2))
    Z_ijs = Z_ijs[neighbourhoods]
    
    for s_i_0 in [ZERO, ONE]:
        marginals[:,s_i_0] = Z_ijs[...,s_i_0].prod(axis=1)
    
    marginals /= marginals.sum(axis=-1,keepdims=True)
    
    
    return marginals


def Z_i_(phis_local, args):

    T = args.T
    d = args.d
    rule = args.rule

    prob_alife = np.zeros([phis_local.shape[0]] + [2] * T + [d + 1] * (T))
    for s_k in product([ZERO, ONE], repeat=T):
        for s_i in product([ZERO, ONE], repeat=T):
            idx_k_to_i = tuple(
                [slice(None), 0] + list(s_k) + list(s_i)
            )
            idx_prob = tuple(
                [slice(None),] + list(s_i) + list(s_k)
            ) 
            prob_alife[idx_prob] += phis_local[idx_k_to_i]

    for k in range(1, d):
        prob_alife__ = np.zeros_like(prob_alife)
        for s_k in product([ZERO, ONE], repeat=T):
            for s_i in product([ZERO, ONE], repeat=T):
                idx_k_to_i = tuple(
                    [slice(None), k] + list(s_k) + list(s_i)
                )

                idx_kp1 = tuple(
                    [slice(None)]
                    + list(s_i)
                    + [slice(s_k[p], k + 1 + s_k[p]) for p in range(T)]
                )  # int(s_k[p] or x_k) the or is the bribe!
                idx_k = tuple(
                    [slice(None)]
                    + list(s_i)
                    + [slice(None, k + 1) for p in range(T)]
                )

                prob_alife__[idx_kp1] += (
                    prob_alife[idx_k]
                    * phis_local[idx_k_to_i][
                        tuple([slice(None)] + [np.newaxis] * T)
                    ]
                )
        prob_alife = prob_alife__

    Z_is = np.zeros(phis_local.shape[0])
    for s_i in product([ZERO, ONE], repeat=T):
        idx_k = tuple(
            [slice(None)]
            + list(s_i)
            + [
                rule(args, s_i_t, s_i_tp1, None)
                for s_i_tp1, s_i_t in zip(list(s_i)[1:], list(s_i)[:-1])
            ]
            + [closing_condition(s_i[-1], s_i[-1], None, args)]
        )
        

        Z_is += prob_alife[idx_k].sum(axis=tuple(range(1, T + 1)))
    return Z_is


def Z_ij_(phi_locals, args):

    T = args.T

    Z_ijs = np.zeros(phi_locals.shape[0])
    for s_j in product([ZERO, ONE], repeat=T):
        for s_i in product([ZERO, ONE], repeat=T):
            idx_i_to_j = tuple(
                [slice(None), 0] + list(s_i) + list(s_j)
            )
            idx_j_to_i = tuple(
                [slice(None), 1] + list(s_j) + list(s_i)
            )

            Z_ijs += (
                phi_locals[idx_i_to_j]
                * phi_locals[idx_j_to_i]
                * (1.0 / args.observables.measure(s_i, s_j)).prod()
            )
    return Z_ijs


def Z_ij_prime_(phi_locals, args):

    T = args.T
    n_observables = args.observables.__len__()

    Z_ijs = np.zeros((phi_locals.shape[0], n_observables))
    for s_j in product([ZERO, ONE], repeat=T):
        for s_i in product([ZERO, ONE], repeat=T):
            idx_i_to_j = tuple(
                [slice(None), 0] + list(s_i) + list(s_j)
            )
            idx_j_to_i = tuple(
                [slice(None), 1] + list(s_j) + list(s_i)
            )
            Z_ijs += (
                phi_locals[idx_i_to_j]
                * phi_locals[idx_j_to_i]
                * 1/ args.observables.measure(s_i, s_j).prod()
            )[:, np.newaxis] * args.observables.values( s_i, s_j)[
                np.newaxis, :
            ]
    return Z_ijs
    
    

class Observables:

    def __init__(self, data, d):

        self.temps = np.array([t for name, t, f in data])
        self.funcs = [f for name, t, f in data]
        self.names = [name for name, t, f in data]
        
        self.d = d

    def values(self, s_i, s_j):
        return np.array([f(s_i, s_j) for f in self.funcs])

    def measure(self,  s_i, s_j):
        values = self.values( s_i, s_j)
        return np.exp(- self.temps / self.d  * values)

    def __len__(self):
        return len(self.temps)
    
    def __str__(self):
        return self.name()

    def name(self, values):
        return {name: v for name, v in zip(self.names, values)}


def normalize(phis):
    # phis: m x ...
    # normalize phi to sum to 1 along all but the first dimension
    return phis / np.sum(phis, axis=tuple(range(1, phis.ndim)), keepdims=True)

def run_BP(args,neighbours_dm1,phi):
    iterations = args.max_iter
    pbar = tqdm(range(iterations))
    converged = False
    for i in pbar:
        phis_local = phi[neighbours_dm1]
        phi_new = compute_phi(phi, phis_local, args)
        
        eps = np.linalg.norm(phi - phi_new)
        phi = (1 - args.damp) * phi + args.damp * phi_new
        if eps < args.eps:
            pbar.close()
            converged = True
            break
        pbar.set_description(f"eps: {eps:.2e}")
    return phi, converged, i

def reaches_consensus_in_T_steps(G,s,T):
    result = evolve_dynamics(G=G,s0=s.astype(bool),T=T)
    return result[-1].mean() == 1.0

def history_passing(args,phi=None, use_wandb=False):
    d = args.d
    T = args.T
    n = args.n

    G = nx.random_regular_graph(d, n, seed=args.seed + 1)
    adj_list = [list(G.neighbors(node)) for node in range(len(G.nodes()))]

    edges = []
    for node in range(len(G.nodes())):
        for neighbor in G.neighbors(node):
            edges.append((node, neighbor))

    edges_array = np.array(edges)

    
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    
    neighbours_dm1 = np.array(
        [[edge_to_idx[(k, i)] for k in adj_list[i] if k != j] for i, j in edges_array]
    )
    neighbours_d = np.array(
        [[edge_to_idx[(i, k)] for k in adj_list[i]] for i in range(len(G.nodes()))] # important that the direction is correct
    )
    
    neighbours_not_unique = np.array(
        [[edge_to_idx[(i, j)], edge_to_idx[(j, i)]] for i, j in edges]
    )
    
    
    rs = np.random.RandomState(args.seed)
    
    phi = rs.uniform(size=tuple([d * n] + [2] * 2 * args.T))
    
    marginals = np.abs(np.random.randn(n,2))
    marginals /= marginals.sum(axis=-1,keepdims=True)
    

    bias = rs.uniform(size=(n,2))
    bias /= bias.sum(axis=-1,keepdims=True)
    
    s_estimate = np.argmax(marginals,axis=-1)
    
    phi = normalize(phi)
    #phi, converged, i = run_BP(args,neighbours_dm1,phi,mask)
    gamma = args.gamma
    prob_update = lambda t:  1- 1/np.power(1+t,gamma)
    t = 0
    bias_new = np.ones((n,2))

    initial_values = []
    final_values = []
    random_values = []
    trajectories = []
    
    evolution = evolve_dynamics(G=G,s0=s_estimate.astype(bool),T=T+25)
    trajectory = [np.mean(evolution[t]) for t in range(len(evolution))]
    trajectory_at_T = trajectory[T]
    
    while t < args.max_its_hp:
        for _ in range(args.bp_its):
            new_shape = tuple([slice(None), slice(None), None] + [None] * (2* (T-1)))
            phi_bias = phi * bias[edges_array[:,0]][new_shape]
            phis_local = phi_bias[neighbours_dm1]
            phi_new = compute_phi(phi_bias, phis_local, args)
            phi = (1 - args.damp) * phi + args.damp * phi_new 
        
        marginals = compute_marginals(phi[neighbours_not_unique], neighbours_d, args)

        bias_new[:,0] = np.where(marginals[:,0] >= marginals[:,1],1-args.bias_param,args.bias_param)
        bias_new[:,1] = 1 - bias_new[:,0]

        bias_update = rs.rand(n) < prob_update(t)
        bias[bias_update] = bias_new[bias_update]

        s_estimate = np.argmax(bias,axis=-1)
        result = evolve_dynamics(G=G,s0=s_estimate.astype(bool),T=T+25)
        result_random = evolve_dynamics(G=G,s0=rs.permutation(s_estimate).astype(bool),T=T)
        trajectory = [np.mean(result[t])*2-1 for t in range(len(result))]
        trajectory_at_T = trajectory[T]

        initial_values.append((s_estimate.mean()*2)-1)
        final_values.append(trajectory_at_T)
        random_values.append((result_random[-1].mean()*2-1))
        trajectories.append(trajectory)
        
        if use_wandb:
            wandb.log({"init":initial_values[-1],"result":final_values[-1], "result_random": random_values[-1]})
        
        t += 1
        

    return initial_values, final_values, random_values, trajectories


def population_dynamics(args,phi=None,use_wandb=False):
    d = args.d
    T = args.T
    n = args.n

    
    rs = np.random.RandomState(args.seed)
    phi_shape = tuple([n] + [2] * 2 * args.T)
    if phi is None:
        phi = np.zeros(phi_shape)#rs.uniform(size=tuple([n] + [2] * 2 * args.T))
    else:
        phi += 0.001 * rs.uniform(size=tuple([n] + [2] * 2 * args.T))
        
    #phi = phi.reshape(-1,4)
    cols = [rs.randint(2, size=(n,)) for _ in range(2 * args.T)]  # Random column indices for each row
    rows = np.arange(n) 
    indices = (rows,) + tuple(cols)
    phi[indices] = 1.0
    #phi = phi.reshape(*phi_shape)
    #phi = normalize(phi)
    

    
    t = 0
    iterations = args.max_iter
    refresh_samples = int(args.damp * n)
    pbar = tqdm(range(iterations))
    converged = False
    n_bins = 100
    eps_interval = 50
    eps = np.inf
    for i in pbar:
        if i % eps_interval == 1:
            #print(phi.shape)
            old_hist =  np.vstack([np.histogram(phi.reshape(-1, 2 ** args.T)[:,i], bins=n_bins, range=(0, 1))[0] for i in range(2 ** args.T)])/n
            
        phi = population_step(args,phi, rs, refresh_samples)
        
        
        if i % eps_interval == 1:
            new_hist =  np.vstack([np.histogram(phi.reshape(-1,2 ** args.T)[:,i], bins=n_bins, range=(0, 1))[0] for i in range(2 ** args.T)])/n
            eps = np.linalg.norm(old_hist - new_hist)
            if use_wandb:
                wandb.log({"eps":eps, 'iteration':i})
            if eps < args.eps:
                pbar.close()
                converged = True
                break
        
        thresholded_phi = (phi >= 0.0001).astype(int).reshape(phi.shape[0],-1)
        #if i > 1000:
        #    phi = normalize(np.where(phi >= 0.0001,phi,0.0))
        unique_rows, counts = np.unique(thresholded_phi, axis=0, return_counts=True)
        sorted_indices = np.lexsort(unique_rows.T[::-1])
        unique_rows_sorted = unique_rows[sorted_indices]
        counts_sorted = counts[sorted_indices]
        
        
        thresh = {
                ''.join(str(bit) for bit in row.flatten()) : count/phi.shape[0] for row, count in zip(unique_rows_sorted,counts_sorted)
            }
        #wandb.log(
        #    thresh
        #)
        
            
        pbar.set_description(f"eps: {eps:.2e}")
    pbar.close()
    norms = phi.sum(axis=tuple(range(1,phi.ndim)))
    print(norms.min(), norms.max())
    rho_temp = args.observables.name(args.observables.temps)['rho_0']
    #np.save(f"results/histograms/final_dist_{T=}_{d=}_rho_temp={rho_temp}",phi.reshape(phi.shape[0],-1))
    
    results = []
    pbar = tqdm(range(args.measurement_its))
    for i in pbar:
        entropy, complexity, observables = population_dynamics_normalization(args,phi, rs)
        phi = population_step(args,phi, rs, refresh_samples)
        results.append(
            {
                "entropy": entropy,
                "complexity": complexity,
                "observables": observables,
            }
        )
        if use_wandb:
            wandb.log({"entropy":entropy, "complexity":complexity, "observables":observables, "measurement_it": i})
        F = phi.reshape(-1,4)
        for i in range(4):
            plt.hist(F[:,i], bins=100,histtype='step')
        plt.yscale('log')
        plt.savefig('s.png')
    pbar.close()     

    df = pd.DataFrame(results)
    observables_avg = (np.array(df.observables.values).mean(axis=0))
    complexity_avg = df.complexity.mean()
    entropy_avg = df.entropy.mean()
    print('observables: ', args.observables.name(observables_avg))
    print('entropy: ', entropy_avg)
    print('complexity OK: ', complexity_avg)
    #if use_wandb:
    #    wandb.log({"observables_avg":observables_avg, "entropy_avg":entropy_avg, "complexity_avg":complexity_avg})
    return {
        "observables": observables_avg,
        "entropy": entropy_avg,
        "complexity": complexity_avg,
        "observables_std": np.array(df.observables.values).std(axis=0),
        "entropy_std": df.entropy.std(),
        "complexity_std": df.complexity.values.std(),
        "observables_all": df.observables.values,
        "entropy_all": df.entropy.values,
        "complexity_all": df.complexity.values
    }
    
def population_step(args,phi, rs, refresh_samples):
    phis_local = phi[rs.randint(0,args.n,size=(refresh_samples,args.d-1))]
    phi_new, Z = compute_phi(phis_local[:,0], phis_local, args,return_Z=True)
    weights = Z.reshape(-1) 
    weights[np.isnan(weights)] = 0.0
    weights = Z.reshape(-1) / Z.sum()
    #print('Z_sum',Z.sum())
    #print('weights',weights.flatten())
    phi_new = phi_new[rs.choice(len(Z), size=refresh_samples, p=weights, replace=True)]
    phi[rs.randint(args.n, size=(refresh_samples,))]=phi_new
    return phi
    
def population_dynamics_normalization(args,phi, rs):
    # get the distribution over the Z_i
    Z_i = 0
    Z_ij = 0
    Z_i_deriv = 0
    Z_ij_deriv = 0

    m = args.m_parisi

    # we calculate Psi(m) here, and we use more samples than during the iterations
    # (a factor of one_measurement samples more)
    one_measurement_samples = int(args.n*args.measurements_factor)

    partial_i_idx = rs.randint(0,phi.shape[0],size=(one_measurement_samples,args.d))

    _i = Z_i_(phi[partial_i_idx], args)

    Z_i += np.power(_i, m).sum()
    Z_i_deriv += (np.power(_i, m) * np.where(_i != 0.0, np.log(_i), 0.0)).sum()

    ij_idx = rs.randint(0,phi.shape[0],size=(one_measurement_samples,2))
    _ij = Z_ij_(phi[ij_idx], args)

    Z_ij += np.power(_ij, m).sum()
    Z_ij_deriv += (np.power(_ij, m) * np.where(_ij != 0.0, np.log(_ij), 0.0)).sum()

    Z_i /= one_measurement_samples
    Z_ij /= one_measurement_samples

    Z_ij_deriv /= one_measurement_samples
    Z_i_deriv /= one_measurement_samples

    
    psi_ = np.log(Z_i) - args.d / 2 * np.log(Z_ij)
    phi_ = Z_i_deriv / Z_i - args.d / 2 * Z_ij_deriv / Z_ij

    observables = 1/Z_ij * (0.5 * Z_ij_prime_(phi[ij_idx], args) * np.where(_ij != 0.0, np.power(_ij,m-1),0.0).reshape(-1,1)).mean(
        axis=0
    )
    
    entropy = phi_ + (args.observables.temps * observables).sum()
    complexity = psi_ - m * phi_
        
    return entropy, complexity, observables


@dataclass
class ConfigPopDyn:
    d : int = 3
    T : int = 3
    rule : Callable = bribe_rule
    loops : str = "homogenous"
    seed : int = np.random.randint(0,100)
    max_iter : int = 1000
    eps : float = 1e-10
    damp : float = 0.05
    observables : Observables = None
    n : int = 200
    rho_temp : float = 0.0
    m_parisi : float = 1.0
    measurement_its : int = 100
    measurements_factor : int = 1.0


@dataclass
class Config:
    d : int = 3
    T : int = 3
    rule : Callable = bribe_rule
    loops : str = "homogenous"
    seed : int = 14  # np.random.randint(0,100)
    max_iter : int = 1000
    eps : float = 1e-10
    damp : float = 0.05
    observables : Observables = None
    n : int = 200
    alpha : float= 0.5 # probability of a node being 1 at initialization (without bribes)
    bp_its : int = 1
    bias_param : int = 0.3
    gamma : int = 0.1
    rho_temp : float = 0.0
    max_its_hp : int = 5000
    break_its_hp: int = 10000
