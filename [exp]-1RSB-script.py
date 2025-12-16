from dataclasses import dataclass
from bdcm.bp_pop_init import *
import numpy as np
from copy import deepcopy as dcp
import pickle
import dataclasses
from pathlib import Path
import wandb
import argparse


tag = "clean-experiments-pop-dyn"

result_dir = Path("results") / tag
result_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse integers n, p, d, and reps.")

    parser.add_argument("--p", type=int, required=True, help="Integer value for p")
    parser.add_argument("--d", type=int, required=True, help="Integer value for d")
    parser.add_argument("--reps", type=int, default=1, help="Integer value for reps")

    args = parser.parse_args()
    
    
    d = args.d # 4
    
    T = args.p # 1
    reps = args.reps # 1
    
    df = pd.read_csv("parameters/temperature_search_d1RSB.csv").set_index(["d","p"])
    row = df.loc[(d,T)]
    
    n = int(row.n) # 1_000_000
    damp = row.damp
    from_temp = row.from_temp
    to_temp = row.to_temp
    steps = int(row.steps)
    for _ in range(reps):
        for rho_temp in np.linspace(from_temp, to_temp, steps):

            seed = np.random.randint(0,30000)

            max_iter = int(row.its)
            eps = 1e-10
            phi = None

            observables = Observables(
                [
                    ("rho_0", rho_temp, lambda s_i, s_j: (s_i[0] + s_j[0])),
                ], d
            )
            write_config = ConfigPopDyn(T=T, d=d, max_iter=max_iter, n=n, seed=seed, eps=eps,damp=damp, rho_temp=rho_temp)
            config = ConfigPopDyn(observables=observables, T=T, d=d, max_iter=max_iter, n=n, seed=seed, eps=eps,damp=damp)
            wandb.init(project="bp-pop-dyn-histograms", entity="feeds", config=write_config)
            results = population_dynamics(config, phi=phi,)
            result_alls = {
                    **results,
                    **dataclasses.asdict(write_config)
            }
            filename = result_dir / f"{d}_{T}_{n}_{rho_temp:.4f}_{seed}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(result_alls, f)
            
            wandb.finish()