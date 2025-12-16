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
            # everything that is larger or equal to half is fine, less if s_j is already zero
            return slice(math.ceil(d / 2), None)
        if s_i_tp1 == ZERO:
            return slice(None, math.floor(d / 2) + 1)  # excluding, check here!

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


def compute_phi(phi, phis_local, args):

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
            
    phi_new = normalize(phi_new)
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

def run_BP(args,neighbours_dm1,phi,mask):
    iterations = args.max_iter
    pbar = tqdm(range(iterations))
    converged = False
    for i in pbar:
        phis_local = phi[neighbours_dm1]
        phi_new = compute_phi(phi, phis_local, args, mask)
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

def history_passing(args,phi=None):
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

    while not reaches_consensus_in_T_steps(G,s_estimate,T) or t < 5000:
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
        result = evolve_dynamics(G=G,s0=s_estimate.astype(bool),T=T)
        result_random = evolve_dynamics(G=G,s0=rs.permutation(s_estimate).astype(bool),T=T)

        print('mean_val',(s_estimate.mean()*2)-1)
        print((result[-1].mean()*2-1), (result_random[-1].mean()*2-1))

        wandb.log({"init":(s_estimate.mean()*2)-1,"result":(result[-1].mean()*2-1), "result_random":(result_random[-1].mean()*2-1)})
        
        t += 1
        
    print("reached solution")



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
