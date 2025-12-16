#SA procedure for finding the strategic initializations leading to global concensus on RRG.

# ---- before running the change of the partial and final results is needed (for different n and d runs) ----

import numpy as np
import networkx as nx
import copy
import time
import os

rng = np.random.default_rng()
#rng = np.random.default_rng(12345)


#we want to store neighboring nodes in a np.array (so nxd array)
def neighbours(G):
    N=np.zeros([n,d])
    for i in range(n):
        count=0
        for k in G.neighbors(i):
            N[i][count]=k
            count+=1
    return N.astype(int)     #.astype(int) convertes float values in the array to int values, which we need because we want to use N values as indices

def onestep_majority(N,s0):
    sums=np.sum(s0[N], axis=1)        #in majority/minority dynamics we just need sums of neighbour values for each node
    return ((1 - np.abs(np.sign(sums)))*s0 + np.sign(sums)) #majority dynamics, always-stay

#returns endstate of node values
def s_endstate(N,s0,p,c):
    for k in range(p+c-1):
        s0=onestep_majority(N,s0)
    return(s0)

def E(N,s0,a,b,p,c):
    s_end=s_endstate(N,s0,p,c)
    return((a*np.sum(s0) - b*np.sum(s_end))/n)

def E_delta(N,s0,a,b,p,c,i):    #E(s')-E(s)
    s_end1=s_endstate(N,s0,p,c)
    s0_copy=copy.deepcopy(s0) #so we still have original s0  
    s0_copy[i]=-s0[i]
    s_end2=s_endstate(N,s0_copy,p,c)
    return((-2*a*s0[i] + b*(np.sum(s_end1) - np.sum(s_end2)))/n)

def m(s):
    return(np.sum(s)/n)

def firstT(N,s0,Tmax):
    ss = copy.deepcopy(s0)
    minit = m(ss)
    if m(ss)==1: return 0,minit
    for t in range(1,Tmax+1):
        ss = onestep_majority(N,ss)
        if m(ss)==1: return t,minit
        if t==Tmax: return Tmax**2, minit

#                                                                                               ---- PARAMETERS ----                                                                              
n_graphs = 10      # number of random graphs
pmax = 3
n = 100
d = 3
c = 1
Tmax = 28
par_a = 1.0005
par_b = 1.0005
save_every = 1     # save after every graph
out_dir = "results_SA_relaxed_p"
os.makedirs(out_dir, exist_ok=True)

# ---- STORAGE ----
t_values = [[None for _ in range(pmax)] for _ in range(n_graphs)]
m_values = [[None for _ in range(pmax)] for _ in range(n_graphs)]


for g in range(n_graphs):
    start_time = time.time()
    print(f"[Graph {g+1}/{n_graphs}]")

    G = nx.random_regular_graph(d, n)
    N = neighbours(G)

    for p in range(1, pmax + 1):
        tts, mms = [], []

        s = 2 * np.random.binomial(n=1, p=0.5, size=[n]) - 1
        a = 0.015 * n                                                                         #initial temperatures setting
        b = 0.01 * n
        t = 0
        m_final = m(s_endstate(N, s, p, c))

        while m_final < 1:
            i = np.random.randint(0, n)
            delta_H = E_delta(N, s, a, b, p, c, i)
            prob_accept = min(1, np.exp(-delta_H))
            if np.random.rand() < prob_accept:
                s[i] = -s[i]
                t_cons, minit = firstT(N, s, Tmax)
                if t_cons <= Tmax:
                    tts.append(t_cons)
                    mms.append(minit)

            if a < 4.5 * n: a *= par_a                                                      #final temperatures setting
            if b < 5.0 * n: b *= par_b

            t += 1
            if t > n**3:
                m_final = 2
            else:
                m_final = m(s_endstate(N, s, p, c))

        t_values[g][p-1] = np.array(tts)
        m_values[g][p-1] = np.array(mms)

        print(f"  finished p={p}, collected {len(tts)} points")

    # ---- SAVE INTERMEDIATE RESULTS ----
    if (g + 1) % save_every == 0:
        save_path = os.path.join(out_dir, f"SA_d3_n100_partial_{g+1}graphs.npz")                                                   #---- file name here ----
        np.savez_compressed(
            save_path,
            **{
                f"t_g{gi}_p{pi+1}": t_values[gi][pi]
                for gi in range(g + 1)
                for pi in range(pmax)
                if t_values[gi][pi] is not None
            },
            **{
                f"m_g{gi}_p{pi+1}": m_values[gi][pi]
                for gi in range(g + 1)
                for pi in range(pmax)
                if m_values[gi][pi] is not None
            }
        )
        print(f"  -> Saved partial results to {save_path}")                

    print(f"[Graph {g+1}] done in {time.time()-start_time:.1f}s\n")

# ---- FINAL SAVE ----
np.savez_compressed(os.path.join(out_dir, "SA_d3_n100_final_results.npz"),                                               #---- and file name here ----
    **{f"t_g{gi}_p{pi+1}": t_values[gi][pi] for gi in range(n_graphs) for pi in range(pmax)},
    **{f"m_g{gi}_p{pi+1}": m_values[gi][pi] for gi in range(n_graphs) for pi in range(pmax)}
)

