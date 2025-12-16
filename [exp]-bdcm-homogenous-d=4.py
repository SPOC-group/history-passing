from config import *
from bdcm.fixed_point_iteration import rs_calculation
from bdcm.fixed_point_iteration import population_dynamics
from bdcm.initialization import init_balanced, init_random_Gauss, init_fixed
from utils.experiments import *
import numpy as np
from copy import deepcopy


c=1
d = 4
for p in [1,2,3]:
        exp_name = f'example_d={d}'
        log_dir = RESULT_DIR / exp_name
        log_dir.mkdir(exist_ok=True, parents=True)

        for r, its,end,N,eps in [('00+11',1000,20,1200,0.0001),
                        ]:
        
        
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


                        dump_pickle({ **res}, log_dir / f'homogenous.its={its}.{r}.{p=}.{c=}.temp={temp:.6f}.pkl')
                        
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
                for temp in np.linspace(0,-end,N):
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


                        dump_pickle({ **res}, log_dir / f'homogenous.its={its}.{r}.{p=}.{c=}.temp={temp:.6f}.pkl')

