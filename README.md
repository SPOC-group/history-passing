This repository contains the code and data required to reproduce the numerical results and figures of the paper:

# Minority Takeover in Majority Dynamics: Searching for Rare Initializations via the History-Passing Algorithm

We study synchronous, deterministic majority dynamics on large random (d)-regular graphs and investigate how much bias in the initial configuration is required to drive the system toward global consensus.

---

## Backtracking Dynamical Cavity Method (BDCM)

**Replica-symmetric results:**
*(to be updated)*

**One-step replica symmetry breaking (1RSB) results:**
*(to be updated)*

**``freezing_evidence``**
--- this folder contains the code used to investigate the freezing in the p=1 case, where the problem can be matched to a constraint satisfaction problem. The code is adapted from *Counting and hardness-of-finding fixed points in cellular automata on random graphs* from the same authors.

---

## History-Passing Reinforcement (HPR)

This section contains the implementation of the **History-Passing Reinforcement (HPR)** algorithm (Algorithm 2 in the paper), designed to explicitly construct strategic initial configurations that lead to consensus despite starting from a global minority.

Code implementing the algorithm together with the associated data can be accessed via the ...

*(Further details to be updated.)*

---

## Simulated Annealing

This repository also includes a simulated annealing (SA) baseline used for comparison with the HPR algorithm.

The SA implementation performs an MCMC search for rare strategic initializations on random regular graphs.

**Relevant files:**

* `SA_RRG.py`
  Core simulated annealing code for finding initial configurations that reach consensus within a prescribed number of time steps p.

* `SA_RRG_relaxed_p_script.py`
  Runs the SA procedure and stores unique pairs of initial magnetization and effective consensus time T_eff, which typically satisfies T_eff > p.

* `SA_time2consensus_plotting.py`
  Generates Figure 15 of the paper using the combined data cache
  `SAdata_cache_combined.pkl`.

The file `SAdata_cache_combined.pkl` contains an aggregated cache of simulation results and is sufficient to reproduce all SA-based figures without rerunning the simulations.

---
