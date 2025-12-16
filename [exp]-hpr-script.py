from dataclasses import dataclass
from bdcm.bp_pop_init import *
import numpy as np
from copy import deepcopy as dcp
import pickle
import dataclasses
from pathlib import Path
import wandb
import argparse
import pandas as pd

use_wandb = False
tag = "clean-experiments-20251103-extra"

result_dir = Path("results") / tag
result_dir.mkdir(exist_ok=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse integers n, p, d, and reps.")

    parser.add_argument("--n", type=int, required=True, help="Integer value for n")
    parser.add_argument("--p", type=int, required=True, help="Integer value for p")
    parser.add_argument("--d", type=int, required=True, help="Integer value for d")
    parser.add_argument("--reps", type=int, required=True, help="Integer value for reps")

    args = parser.parse_args()
    
    
    d = args.d 
    n = args.n
    T = args.p
    reps = args.reps
    
    df = pd.read_csv("parameters/HPR_hyperparameters.csv").set_index(["d","p","n"])
    row = df.loc[(d,T,n)]
    
    bias_param = row.bias
    rho_temp = row.temp
    damp = row.damp
    

    max_iter = 100
    eps = 1e-10
    phi = None
    bp_its = 1
    max_its_hp = 10000
    
    
    for _ in range(reps):
        seed = np.random.randint(0,30000)
        observables = Observables(
            [
                ("rho_0", rho_temp, lambda s_i, s_j: s_i[0] * d ),
                #("rho_0", rho_temp, lambda x_i, x_j, s_i, s_j:  (2 * s_i[0] - 1)* d),
                ("rho_final", 0.0, lambda s_i, s_j: s_i[-1] + s_j[-1]),
            ]
            + [
                (f"rho_{i}", 0.0, lambda s_i, s_j: s_i[i] + s_j[i]) for i in range(1, T)
            ], d
        )

        config = Config(observables=observables, T=T, d=d, max_iter=max_iter, n=n, seed=seed, eps=eps,damp=damp, bp_its=bp_its, bias_param=bias_param,max_its_hp=max_its_hp)
        write_config = Config(T=T, d=d, max_iter=max_iter, n=n, seed=seed,eps=eps,damp=damp, bp_its=bp_its, bias_param=bias_param, rho_temp=rho_temp, max_its_hp=max_its_hp)
        if use_wandb:
            wandb.init(project="bp", entity="feeds", config=write_config)

        initial_values, final_values, random_values, trajectories = history_passing(config, phi=phi, use_wandb=use_wandb)
        results = {
            'initial_mag': initial_values,
            'final_mag': final_values,
            'random_init_mag': random_values,
            'trajectories': trajectories,
            **dataclasses.asdict(write_config)
        }
        filename = result_dir / f"{d}_{T}_{n}_{seed}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        if use_wandb:
            wandb.finish()