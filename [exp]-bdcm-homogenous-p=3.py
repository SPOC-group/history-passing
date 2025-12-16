from config import *
from bdcm.fixed_point_iteration import rs_calculation
from bdcm.fixed_point_iteration import population_dynamics
from bdcm.initialization import init_balanced, init_random_Gauss, init_fixed
from utils.experiments import *
import numpy as np
from copy import deepcopy


c=1
p=3
exp_name = f'example_p={p}_final'
log_dir = RESULT_DIR / exp_name
log_dir.mkdir(exist_ok=True, parents=True)



experiments = [('0+1',20000,30,1200,0.0001),
                        ('0011',20000,7,1200,0.0001),
                     ('00+11',20000,7,1200,0.0001),
                     ('000111',40000,7,1200, 0.001),]
experiments = [
                     ('000+111',40000,7,1200, 0.001),
                     ('00001111',20000,7,1200,0.001),
                     ('0000+1111',20000,7,1200,0.001),
                     ('0000011111',20000,7,1200,0.001),]

max_temp = 18
experiments = []
experiments += [('0' * i + '1' * i, 40000,  max_temp, 2400, 0.001) for i in range(6,30)]
experiments += [('0' * i + '+' * 1 + '1' * i, 40000, max_temp, 2400, 0.001) for i in range(6,30)[::2]]

for r, its,end,N,eps in experiments:
    res = rs_calculation(its=its, 
                alpha=0.99, 
                rule_code=r, 
                c=c, 
                d=len(r) - 1,
                        log_dir=log_dir, 
                        log_suffix=f'end',
                        mag_init_temp=0.0,
                        p=p,
                        homogenous_point_attractor=1,
                        fix_observable=None,
                        init_func=init_random_Gauss,
                        epsilon=eps
                        )

    chi = res['chi']
    for temp in np.linspace(0,end,N):
        res = rs_calculation(its=its, 
                alpha=0.99, 
                rule_code=r, 
                c=c, 
                d=len(r) - 1,
                        log_dir=log_dir, 
                        log_suffix=f'end',
                        mag_init_temp=temp,
                        p=p,
                        homogenous_point_attractor=1,
                        fix_observable=None,
                        init_func=lambda shape: init_fixed(shape,res['chi']),
                        epsilon=eps
                        )
        print('**********')
        print('RS entropy', res['entropy'])
        print('**********')
        
        chi = res['chi']


        dump_pickle({ **res}, log_dir / f'homogenous.its={its}.{np.random.randint(3000)}.{p=}.{c=}.temp={temp:.6f}.pkl')

