This repository contains the code and data required to reproduce the numerical results and figures of the paper:

# Minority Takeover in Majority Dynamics: Searching for Rare Initializations via the History-Passing Algorithm

We study synchronous, deterministic majority dynamics on large random (d)-regular graphs and investigate how much bias in the initial configuration is required to drive the system toward global consensus.

---

## Backtracking Dynamical Cavity Method (BDCM)

Folder ``bdcm`` contains the implementation of the BDCM and all associated function definitions. These modules can be imported to generate the BDCM results (both *symmetric* and *replica symmetry broken*) using the provided scripts.

* ``[figures]-d=4-overview.ipynb`` (left), ``[figures]-static-1RSB.ipynb`` (right) produce Figure 1.

* ``[figures]-bdcm.ipynb``, ``[figures]-rs-p=1-extrapolations.ipynb`` lead to Table 1 and Figure 2.

* m_sample data in Tables 2, 3 come partialy from `[1]` and ``[figures]-d=6-c=2.ipynb`` (code used also for creating Figure 6). 

* ``[figures]-static-1RSB.ipynb`` gives Figures 8, 9.

**``freezing_evidence``**
--- this folder contains the code used to investigate the freezing in the p=1 case, where the problem can be matched to a constraint satisfaction problem. The code is adapted from *Counting and hardness-of-finding fixed points in cellular automata on random graphs* from the same authors.

> `[1]` Freya Behrens, Barbora Hudcova, and Lenka Zdeborova. Dynamical phase transitions in graph cellular
automata. Phys. Rev. E, 109(4):044312, 2024
---

## History-Passing Reinforcement (HPR)
HPR is designed to construct strategic initial configurations that lead to consensus despite starting from a global minority for a given graph.

The folder ``bdcm`` contains ``bp_pop_init.py``, where the implementation of the HPR algorithm (Algorithm 2) on RRGs together with the population dynamics procedure can be found. The associated data and code for the recreation of Figures 12, 13, 14 can be accessed in ``[figures]-hpr.ipynb``.

The folder ``parameters`` contains hyperparameters used for HPR (and d1RSB) results. 

---

## Simulated Annealing

This repository also includes a simulated annealing (SA) baseline used for comparison with the HPR algorithm.

The SA implementation performs an MCMC search for rare strategic initializations on random regular graphs.

* `SA_RRG.py`
  Core simulated annealing code for finding initial configurations that reach consensus within a prescribed number of time steps p.

* `SA_RRG_relaxed_p_script.py`
  Runs the SA procedure and stores unique pairs of initial magnetization and effective consensus time T_eff, which typically satisfies T_eff > p.

* `SA_time2consensus_plotting.py`
  Generates Figure 15 of the paper using the combined data cache
  `SAdata_cache_combined.pkl`.

The file `SAdata_cache_combined.pkl` contains an aggregated cache of simulation results and is sufficient to reproduce all SA-based figures without rerunning the simulations.

---
