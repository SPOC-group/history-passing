import numpy as np
import networkx as nx
from tqdm import tqdm
import math

ONE = 1
ZERO = 0

def onestep_majority(N,s0):
    #print('s0',s0)
    sums=np.sum(s0[N], axis=1)   
    #print(sums)#in majority/minority dynamics we just need sums of neighbour values for each node
    return ((1 - np.abs(np.sign(sums)))*s0 + np.sign(sums)) #majority dynamics, always-stay

def evolve_dynamics_new(G,s0,x,T):
    
    s = evolve_dynamics_old(G,s0.copy(),x,T)
    
    
    N = np.array([list(G.neighbors(i)) for i in range(G.number_of_nodes())])
    
    s0_ = (s0 * 2) - 1
    
    for t in range(T):
        #print('before __ ', s0_)
        s0_ = onestep_majority(N,s0_)
        #print('after __ ', s0_)
        
    
    s0_ = (s0_ + 1) / 2
    
    #print(s0_.astype(int),np.array(s[-1]).astype(int))
    
    assert np.allclose(s0_.astype(int),np.array(s[-1]).astype(int))
    
    return [s0_.astype(int)]
    

def evolve_dynamics(G,s0,x,T):
    n = G.number_of_nodes()
    
    edges = np.array(G.edges())
    edges = np.concatenate([edges, edges[:, [1, 0]]])

    # get all neighbours of the node i
    neighbours = [set() for i in range(n)]
    for i, j in edges:
        neighbours[i].add(j)
        neighbours[j].add(i)
        
    neighbours = np.array([list(l) for l in neighbours])
    
    d = len(neighbours[0])
    half = d / 2

    s = [s0]
    for t in range(T):
        s_prev = s[-1]
        neigh_cnt = (s_prev)[neighbours].sum(axis=1)
        st = np.where(neigh_cnt > d - half, ONE, np.where(neigh_cnt < half, ZERO, s_prev))
        s.append(st)
        
    return s

def init_biased(n,p_bias,rs=np.random):
        # p_bias will be one
        
        n_bias = math.floor(n * p_bias)
        v = np.concatenate( (np.ones(n_bias), np.zeros(n - n_bias))).astype(int)
        rs.shuffle(v)
        
        return v.astype(int)
    