import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from bdcm.rules import TotalisiticRule, DEAD, ALIFE, X_POS, Y_POS
from bdcm.initialization import init_random_Gauss
from bdcm.observables import Observables
from bdcm.utils import sample_from_cdf
opposite = lambda x: 0 if x == 1 else 1

def Z_i_(chis, d, allowed_alife_neighbours, attr_size_graph, attractor_graph,homogenous_point_attractor,require_full_rattling,all_rattlers):
    if len(chis.shape) == (attr_size_graph * 2):
        chis = [chis for _ in range(d)]
    Z = 0
    shape = [0] * (attr_size_graph * 2)
    shape[:attr_size_graph] = [2] * attr_size_graph
    shape[attr_size_graph:] = [d + 1] * attr_size_graph
    ALL_ALLOWED = np.arange(d+1)
    prob_alive = np.zeros(shape)
    # what is the probability of having k_1 alife neighbours in period p_1 and k_2 in p_2 if x is dead? prob_alive[DEAD, k1,k2]
    for c in product([DEAD, ALIFE], repeat=attr_size_graph):
        for x in product([DEAD, ALIFE], repeat=attr_size_graph):
            idx = [0] * attr_size_graph * 2
            idx[X_POS::2] = c
            idx[Y_POS::2] = x
            idx = tuple(idx)

            p_idx = [0] * (attr_size_graph * 2)
            p_idx[:attr_size_graph] = x
            p_idx[attr_size_graph:] = c
            p_idx = tuple(p_idx)
            prob_alive[p_idx] += chis[0][idx]
    for k in range(1, d):
        prob_alive__ = np.zeros_like(prob_alive)
        for c in product([DEAD, ALIFE], repeat=attr_size_graph):
            for x in product([DEAD, ALIFE], repeat=attr_size_graph):
                idx = [0] * attr_size_graph * 2
                idx[X_POS::2] = c
                idx[Y_POS::2] = x
                idx = tuple(idx)

                p_idx = [0] * (attr_size_graph * 2)
                p_idx[:attr_size_graph] = x
                p_idx[attr_size_graph:] = [slice(c[p], k + 1 + c[p]) for p in range(attr_size_graph)]
                p_idx = tuple(p_idx)

                o_idx = [0] * (attr_size_graph * 2)
                o_idx[:attr_size_graph] = x
                o_idx[attr_size_graph:] = [slice(None, k + 1) for p in range(attr_size_graph)]
                o_idx = tuple(o_idx)

                prob_alive__[p_idx] += prob_alive[o_idx] * chis[k][idx]
        prob_alive = prob_alive__
        

    for c in product([DEAD, ALIFE], repeat=attr_size_graph):
        allowals = []

        if all_rattlers is not None:
            for curr, nex in enumerate(attractor_graph[:-1]):
                allowed, change_occured = check_Nx_allowance_Z(c, curr, nex, allowed_alife_neighbours, d)
                allowals.append(allowed)
            current_OUT = c[-1]
            next_OUT = 0 if c[-1] == 1 else 1

            allowed = allowed_alife_neighbours[current_OUT]

            # if change occurs, we need to change the neighbour
            change_occurs = current_OUT != next_OUT
            if change_occurs:
                # we actually want the node to change their color, so we need to make them unahppy in this step!
                allowed = not_(allowed, d + 1)

            allowals.append(allowed)
        elif homogenous_point_attractor is not None:
            for curr, nex in enumerate(attractor_graph[:-1]):
                allowed,change_occured = check_Nx_allowance_Z(c, curr, nex, allowed_alife_neighbours, d)
                allowals.append(allowed)
            current_OUT = c[-1]
            next_OUT = homogenous_point_attractor

            allowed = allowed_alife_neighbours[current_OUT]

            # if change occurs, we need to change the neighbour
            change_occurs = current_OUT != next_OUT
            if change_occurs:
                # we actually want the node to change their color, so we need to make them unahppy in this step!
                allowed = not_(allowed, d + 1)

            allowals.append(allowed)
        else:
            for curr, nex in enumerate(attractor_graph):
                if nex is None:
                    allowals.append(ALL_ALLOWED)
                    continue
                allowed,change_occured = check_Nx_allowance_Z(c, curr, nex, allowed_alife_neighbours, d)
                if curr == len(attractor_graph) - 1 and require_full_rattling and not change_occured:
                    allowals.append([])
                else:
                    allowals.append(allowed)

        Z += prob_alive[tuple(c)][np.ix_(*allowals)].sum()
    return Z

def check_Nx_allowance_Z(node, current_step, next_step, allowed_alife_neighbours, d):
    current_OUT = node[current_step]
    next_OUT = node[next_step]

    allowed = allowed_alife_neighbours[current_OUT]

    # if change occurs, we need to change the neighbour
    change_occurs = current_OUT != next_OUT
    if change_occurs:
        # we actually want the node to change their color, so we need to make them unahppy in this step!
        allowed = not_(allowed, d + 1)

    return allowed, change_occurs


def not_(allowed, d):
    return np.delete(np.arange(d), allowed)

from functools import lru_cache



def get_best_mu_field(chi_hat, rho, mu0, m, d, attr_size_graph, last_mu=None):
    def f(mu):
        return (m - get_M(mu, rho,mu0, chi_hat, d, attr_size_graph)) ** 2

    best = fmin(f, 0.0 if last_mu is None else last_mu, disp=False, ftol=0.0000001, xtol=0.0000001)[0]
    return best
from tqdm import tqdm
# raise Exception

# message_(X->Y)




def rs_calculation(its, rule_code, c=1, p=0,
                   init_func=init_random_Gauss, alpha=0.99,
                   m_target=None, attraction_factor=2, fix_observable=None, balance_colors=False,
                   log_dir=None, log_suffix='', all_rattlers=None, homogenous_point_attractor=None, require_full_rattling=False, epsilon=None, target_observable=None, target_value=None,**kwargs):
    """

        :param its: # of iterations
        :param rule_code: code of the rule
        :param c: length of limit cycle. If None or 0, then no limit cycle is assumed
        :param p: length of incoming path
        :param init_func: function to initialize the configuration
        :param alpha: dampening factor
        :param m_target:
        :param attraction_factor:
        :param fix_observable:
        :param balance_colors:
        :param log_dir:
        :param log_suffix:
        :param all_rattlers:
        :param homogenous_point_attractor:
        :param require_full_rattling:
        :param epsilon: convergence threshold. If None, then no convergence is checked but it is run until max iterations
        :param kwargs:
        :return:
    """

    d = len(rule_code) - 1

    if c == 0:
        raise ValueError("Cycle length c must be >0 or None.")

    if homogenous_point_attractor is not None:
        assert homogenous_point_attractor in [0,1]
        assert c == 1
        c = 0

    if all_rattlers is not None:
        assert c == 2
        c = 1

    has_cycle = c

    # define the length of the constraints
    # we still need to add +1 for the final state if there is no cycle
    attr_size_graph = (c if c is not None else 1) + p


    # define the configuration file
    config = {**locals(), **kwargs}
    del config['kwargs']
    config['init_func'] = config['init_func'].__name__

    # the real processing starts here

    ALL_ALLOWED = np.arange(d)


    rule_code = TotalisiticRule(rule_code, d)
    allowed_alife_neighbours = rule_code.allowed_alife_neighbours
    observables = Observables(**config)

    attractor_graph = [(p +1) % attr_size_graph for p in range(attr_size_graph)]
    attractor_graph[-1] = p if has_cycle is not None else None # also allow to not have cycles


    if epsilon is None:
        # stick to iterations and do not break on convergence
        epsilon = 10e-20

    if fix_observable is not None:
        target_observable, target_value = fix_observable


    shape = tuple([2] * (attr_size_graph * 2))
    # this is the BP message
    # it is an array of shape (2,2,...2) where the number of 2s is the number of nodes in the attractor graph,
    # i.e. the length of the overall time 2(p+c)
    # the entries are defined as follows:
    # chi_(X->Y)[x^1,y^1,x^2,y^2,...,x^(p+c),y^(p+c)]
    #
    # so if we want to create an index idx to that array, where the values of X andY are
    # X = [x^1,x^2,...,x^(p+c)]
    # Y = [y^1,y^2,...,y^(p+c)]
    # then we can do it as follows:
    # idx = [None] * 2 * (p+c)
    # idx[X_node::2] = X
    # idx[Y_node::2] = Y
    # then we can use
    # chi[tuple(idx)]


    chi = init_func(shape)
    #chi[:] = 0
    #chi[1,1,1,1] = 1
    chi /= chi.sum()

    # track the initial state
    init_chi = chi.copy()

    # create function helper that is cached
    @lru_cache(maxsize=None)
    def determine_allowals(current_out, current_in, next_out):
        # outgoing node -> incoming node
        allowed = allowed_alife_neighbours[current_out] - int(current_in == ALIFE)
        allowed = allowed[allowed <= d - 1]
        allowed = allowed[allowed >= 0]

        # if change occurs, we need to change the neighbour
        change_occurs = current_out != next_out
        if change_occurs:
            # we actually want the node to change their color, so we need to make them unahppy in this step!
            allowed = not_(allowed, d)

        return allowed, change_occurs

    # this iterated the BP
    # We have a node CENTER, which is the center of a neighbourhood of d nodes
    # Then, d-1 messages are coming into the center node, from nodes which we call PARENTS
    # Finally, the CENTER node sends exactly 1 message to its child
    # PARENT_1 \
    # PARENT_2 -> CENTER -> CHILD
    # PARENT_3 /
    # There is also the situation when we iterate over all d nodes and do not distinguish CHILDREN and PARENTS
    # - then we call all neighbouring nodes the NEIGHBOURS

    i = 0
    converged = False
    while i < its and not converged:
        i+=1

        chi_dp, _ = step_rs(ALL_ALLOWED, all_rattlers, attr_size_graph, attractor_graph, balance_colors, chi, d,
                      determine_allowals, fix_observable, homogenous_point_attractor, observables, target_observable,
                      target_value)

        chi__ = chi_dp.copy()
        chi__ = chi__ / chi__.sum()
        chi_old = chi.copy()
        chi = alpha * chi + (1 - alpha) * chi__
        dist = abs(chi_old - chi__).sum()

        #if i % 1000 == 1:
        #    print(chi.flatten())
            
        if dist < epsilon:
            #print(f'conveged early at it={i}')
            converged = True

    #print('convergence', abs(chi_old - chi__).sum())
    #print("... done fixed


    Z_i = Z_i_(chi, d, allowed_alife_neighbours, attr_size_graph, attractor_graph,homogenous_point_attractor,require_full_rattling,all_rattlers)

    observed = observables.calc_marginals(chi)
    """
    print('Z_i', Z_i)
    print('Z_ij', observed['Z_ij'])
    print('log(Z_i)', np.log(Z_i)) 
    print('log(Z_ij)', np.log(observed['Z_ij']))
    print('Legendre', observed['Legendre'])
    print('mag_init', observed['mag_init'])
    """
    Phi_RS = np.log(Z_i) - d / 2 * np.log(observed['Z_ij'])

    entropy = Phi_RS + observed['Legendre']

    #print('\n'.join([str(a) for a in zip(np.argwhere(chi < 100), chi.flatten())]))

    return {
        **observed,
        **config,
        'init_chi': init_chi,
        'Phi_RS': Phi_RS,
        'entropy': entropy,
        'chi': chi,
        'converged': converged,
        'accuracy': abs(chi_old - chi__).sum()
    }

def step_rs(ALL_ALLOWED, all_rattlers, attr_size_graph, attractor_graph, balance_colors, chi, d, determine_allowals,
            fix_observable, homogenous_point_attractor, observables, target_observable, target_value):

    # we may want to fix the observable to a value directly, instead of fixing the temperature
    # this routine adapts the temperature based on the target observable and the targetted value
    if fix_observable is not None:
        observables.update_best_temps(target_observable, target_value, chi)
    chis = [chi for _ in range(d-1)]

    return step(ALL_ALLOWED, all_rattlers, attr_size_graph, attractor_graph, balance_colors, chis, d, determine_allowals,
         homogenous_point_attractor, observables)


def step(ALL_ALLOWED, all_rattlers, attr_size_graph, attractor_graph, balance_colors, chis, d, determine_allowals,
         homogenous_point_attractor, observables):

    shape = tuple([2] * (attr_size_graph * 2))
    chi_dp = np.zeros(shape)
    shape = [0] * (attr_size_graph * 2)
    shape[:attr_size_graph] = [2] * attr_size_graph  # first part is the state of the center node
    shape[attr_size_graph:] = [d] * attr_size_graph  # second part is the number of alive parents at every time step in the neighbourhood
    # the probability we do the dynamic programming over here, is defined as follows:
    # prob[center^1,center^2,...,center^(p+c),k_1,k_2,...,k_(p+c)]
    # where center^i is the state of the center node at time i
    # and k_i is the number of alive parents at time i
    prob = np.zeros(shape)
    # STEP Initialize the DP probability calculation
    for PARENT in product([DEAD, ALIFE], repeat=attr_size_graph):
        for CENTER in product([DEAD, ALIFE], repeat=attr_size_graph):
            idx = [0] * attr_size_graph * 2
            idx[X_POS::2] = PARENT
            idx[Y_POS::2] = CENTER
            idx = tuple(idx)

            p_idx = [0] * (attr_size_graph * 2)
            p_idx[:attr_size_graph] = CENTER
            p_idx[attr_size_graph:] = PARENT
            p_idx = tuple(p_idx)
            prob[p_idx] += chis[0][idx]
    #print(prob)
    
    # STEP Do the DP, this does not incorporate any constraints on the cycles
    for k in range(1, d - 1):
        prob_alive__ = np.zeros_like(prob)
        for PARENT in product([DEAD, ALIFE], repeat=attr_size_graph):
            for CENTER in product([DEAD, ALIFE], repeat=attr_size_graph):
                idx = [0] * attr_size_graph * 2
                idx[X_POS::2] = PARENT
                idx[Y_POS::2] = CENTER
                idx = tuple(idx)

                p_idx = [0] * (attr_size_graph * 2)
                p_idx[:attr_size_graph] = CENTER
                p_idx[attr_size_graph:] = [slice(PARENT[p], k + 1 + PARENT[p]) for p in range(attr_size_graph)]
                p_idx = tuple(p_idx)

                o_idx = [0] * (attr_size_graph * 2)
                o_idx[:attr_size_graph] = CENTER

                o_idx[attr_size_graph:] = [slice(None, k + 1) for p in range(attr_size_graph)]
                o_idx = tuple(o_idx)

                prob_alive__[p_idx] += prob[o_idx] * chis[k][idx]
        prob = prob_alive__
    
    #print(prob)
    
        # Step to sum only over the allowed combinations for each chi, this changes depending on whether we care about cycles, or cycles with paths as attractors
    for CENTER in product([DEAD, ALIFE], repeat=attr_size_graph):
        for CHILD in product([DEAD, ALIFE], repeat=attr_size_graph):
            allowals = []

            if homogenous_point_attractor is not None:
                for t, tp1 in enumerate(attractor_graph[:-1]):  # ensure consistency with the attractor graph
                    allowed, change_occured = determine_allowals(CENTER[t], CHILD[t], CENTER[tp1])
                    allowals.append(allowed)

                t = len(attractor_graph) - 1
                # for the last state, check that it goes into [1,1]
                allowed, change_occured = determine_allowals(CENTER[t], CHILD[t], homogenous_point_attractor)
                allowals.append(allowed)

            elif all_rattlers is not None:
                for t, tp1 in enumerate(attractor_graph[:-1]):
                    allowed, change_occured = determine_allowals(CENTER[t], CHILD[t], CENTER[tp1])
                    allowals.append(allowed)

                t = len(attractor_graph) - 1
                # for the last state, check that it goes into [1,1]
                allowed, change_occured = determine_allowals(CENTER[t], CHILD[t], opposite(CENTER[t]))
                allowals.append(allowed)

            else:
                for t, tp1 in enumerate(attractor_graph):
                    if tp1 is None:
                        allowals.append(ALL_ALLOWED)
                    else:
                        allowed, change_occured = determine_allowals(CENTER[t], CHILD[t], CENTER[tp1])
                        allowals.append(allowed)

            idx = [0] * attr_size_graph * 2
            idx[X_POS::2] = CENTER
            idx[Y_POS::2] = CHILD
            idx = tuple(idx)
            chi_dp[idx] = prob[tuple(CENTER)][np.ix_(*allowals)].sum()
    #print(chi_dp)
    
    # STEP to append the priors to the distribution
    for CENTER in product([DEAD, ALIFE], repeat=attr_size_graph):
        for CHILD in product([DEAD, ALIFE], repeat=attr_size_graph):
            PARENT = [0] * (attr_size_graph * 2)
            PARENT[X_POS::2] = CENTER
            PARENT[Y_POS::2] = CHILD
            #print(CENTER,CHILD,observables.g(CENTER, CHILD))
            chi_dp[tuple(PARENT)] *= observables.g(CENTER, CHILD).prod()
    if balance_colors:
        for NEIGHBORS in product([DEAD, ALIFE], repeat=2 * attr_size_graph):
            # set the inputs such that the inverse of every item is equal to its color symmetric item
            SYM_NODES = tuple(0 if kk == 1 else 1 for kk in NEIGHBORS)
            temp = (chi_dp[tuple(NEIGHBORS)] + chi_dp[SYM_NODES]) / 2
            chi_dp[tuple(NEIGHBORS)] = temp
            chi_dp[tuple(SYM_NODES)] = temp
    
    Z_ij = chi_dp.sum()
    return chi_dp / Z_ij, Z_ij

from bdcm.population_dynamics import step as pop_step
def population_dynamics(chi_comp,its, rule_code, c=1, p=0, init_func=init_random_Gauss, alpha=0.99,m_parisi = 0.0, m_target=None, population_n=600, attraction_factor=2, fix_observable=None, balance_colors=False,
log_dir=None, log_suffix='', all_rattlers=None, homogenous_point_attractor=None, require_full_rattling=False, epsilon=None, target_observable=None, target_value=None,**kwargs):
    """

        :param its: # of iterations
        :param rule_code: code of the rule
        :param c: length of limit cycle. If None or 0, then no limit cycle is assumed
        :param p: length of incoming path
        :param init_func: function to initialize the configuration
        :param alpha: dampening factor
        :param m_target:
        :param attraction_factor:
        :param fix_observable:
        :param balance_colors:
        :param log_dir:
        :param log_suffix:
        :param all_rattlers:
        :param homogenous_point_attractor:
        :param require_full_rattling:
        :param epsilon: convergence threshold. If None, then no convergence is checked but it is run until max iterations
        :param kwargs:
        :return:
    """

    d = len(rule_code) - 1

    if c == 0:
        raise ValueError("Cycle length c must be >0 or None.")

    if homogenous_point_attractor is not None:
        assert homogenous_point_attractor in [0,1]
        assert c == 1
        c = 0

    if all_rattlers is not None:
        assert c == 2
        c = 1

    has_cycle = c

    # define the length of the constraints
    # we still need to add +1 for the final state if there is no cycle
    attr_size_graph = (c if c is not None else 1) + p


    # define the configuration file
    config = {**locals(), **kwargs}
    del config['kwargs']
    config['init_func'] = config['init_func'].__name__

    log_file = log_dir / f'pop.its={its}.{rule_code}.{p=}.{c=}.{m_parisi=}.ENTROPY.csv'

    # the real processing starts here

    ALL_ALLOWED = np.arange(d)


    rule_code = TotalisiticRule(rule_code, d)
    allowed_alife_neighbours = rule_code.allowed_alife_neighbours
    observables = Observables(**config)

    attractor_graph = [(p +1) % attr_size_graph for p in range(attr_size_graph)]
    attractor_graph[-1] = p if has_cycle is not None else None # also allow to not have cycles


    if epsilon is None:
        # stick to iterations and do not break on convergence
        epsilon = 10e-20

    if fix_observable is not None:
        target_observable, target_value = fix_observable


    shape = tuple([2] * (attr_size_graph * 2))
    # this is the BP message
    # it is an array of shape (2,2,...2) where the number of 2s is the number of nodes in the attractor graph,
    # i.e. the length of the overall time 2(p+c)
    # the entries are defined as follows:
    # chi_(X->Y)[x^1,y^1,x^2,y^2,...,x^(p+c),y^(p+c)]
    #
    # so if we want to create an index idx to that array, where the values of X andY are
    # X = [x^1,x^2,...,x^(p+c)]
    # Y = [y^1,y^2,...,y^(p+c)]
    # then we can do it as follows:
    # idx = [None] * 2 * (p+c)
    # idx[X_node::2] = X
    # idx[Y_node::2] = Y
    # then we can use
    # chi[tuple(idx)]


    chis = [init_func(shape) for _ in range(population_n)]
    chis = [chi / chi.sum() for chi in chis]


    # track the initial state
    init_chi = None

    # create function helper that is cached
    @lru_cache(maxsize=None)
    def determine_allowals(current_out, current_in, next_out):
        # outgoing node -> incoming node
        allowed = allowed_alife_neighbours[current_out] - int(current_in == ALIFE)
        allowed = allowed[allowed <= d - 1]
        allowed = allowed[allowed >= 0]

        # if change occurs, we need to change the neighbour
        change_occurs = current_out != next_out
        if change_occurs:
            # we actually want the node to change their color, so we need to make them unahppy in this step!
            allowed = not_(allowed, d)

        return allowed, change_occurs

    # this iterated the BP
    # We have a node CENTER, which is the center of a neighbourhood of d nodes
    # Then, d-1 messages are coming into the center node, from nodes which we call PARENTS
    # Finally, the CENTER node sends exactly 1 message to its child
    # PARENT_1 \
    # PARENT_2 -> CENTER -> CHILD
    # PARENT_3 /
    # There is also the situation when we iterate over all d nodes and do not distinguish CHILDREN and PARENTS
    # - then we call all neighbouring nodes the NEIGHBOURS
    dist = []
    rs = np.random.RandomState(42)
    i = 0
    converged = False
    Phi_entropies = []
    complexities = []
    s_internals = []
    while i < its and not converged:
        i+=1
        chis = np.array(chis)

        kk = chis.reshape(population_n, -1)
        dist.append(np.abs(kk - chi_comp.flatten()).mean())
        if False and i % 100 == 0:
            for k in range(kk.shape[1]):
                plt.hist(kk[:, k], histtype='step')
            # plt.title(chis.mean(axis=0).flatten())
            plt.title(f'{i=}, p={p}')
            plt.xlim(0, 0.5)
            plt.show()
            plt.plot(dist)
            plt.axhline(0.0)
            plt.show()
        alp = 0.1
        update_mask = np.random.choice([True, False], p=[alp, 1 - alp], size=(population_n))
        update_n = update_mask.sum()
        neighbour_chis = chis[rs.randint(population_n, size=(update_n, d - 1))]

        new_step = [step(ALL_ALLOWED, all_rattlers, attr_size_graph, attractor_graph, balance_colors, neighbours, d, determine_allowals,
         homogenous_point_attractor, observables) for neighbours in
                neighbour_chis]
        new_chis = [chi for chi, Z_i_to_j in new_step]
        Z_i_to_js = [Z_i_to_j for chi, Z_i_to_j in new_step]


        if m_parisi != 0.0:
            # if Z_i_to_j is zero, then we cannot reweight with np.nan, so we need to remove those
            new_chis = [chi for chi, Z_i_to_j in new_step if Z_i_to_j > 0.0]
            Z_i_to_js = [Z_i_to_j for chi, Z_i_to_j in new_step if Z_i_to_j > 0.0]

            if len(new_chis) == 0:
                print('***could not reweight***')
                continue


            weights = np.power(Z_i_to_js, m_parisi, dtype=np.float64)
            reweighted_idx = sample_from_cdf(rs, cdf=weights.cumsum() / weights.sum(), n_samples=update_n)
            new_chis = np.array(new_chis)
            new_chis = new_chis[reweighted_idx]
        new_chis = np.array(new_chis)
        chis[update_mask] = new_chis
        if i>50 and i % 100 == 0:
            Phi_entropy, complexity, s = compute_Phi_sigma_s(chis, d, m_parisi, observables, population_n, allowed_alife_neighbours, attr_size_graph,
                                attractor_graph,
                                homogenous_point_attractor, require_full_rattling, all_rattlers)
            #print(Phi_entropy, complexity, s)
            Phi_entropies.append(Phi_entropy)
            complexities.append(complexity)
            s_internals.append(s)
            with open(log_file, 'a') as file:
                file.write(f'{i}, {Phi_entropy}, {complexity}, {s}\n')
    print("... done fixed point iteration.")

    Phi_entropy, complexity, s = compute_Phi_sigma_s(chis, d, m_parisi, observables, population_n,allowed_alife_neighbours, attr_size_graph, attractor_graph,
                             homogenous_point_attractor, require_full_rattling, all_rattlers)
    Phi_entropies.append(Phi_entropy)
    complexities.append(complexity)
    s_internals.append(s)

    return {
        **config,
        'init_chi': init_chi,
        'Phi_entropy': np.mean(Phi_entropies),
        'complexity': np.mean(complexities),
        's_internal': np.mean(s_internals),
        's_internals': s_internals,
        'Phi_entropies': Phi_entropies,
        'complexities': complexities,
        's_total': s + Phi_entropy if s + Phi_entropy > 0 else -np.inf ,
        'converged': converged
    }


def compute_Phi_sigma_s(chis, d, m_parisi, observables, population_n,allowed_alife_neighbours, attr_size_graph, attractor_graph,
                             homogenous_point_attractor, require_full_rattling, all_rattlers):
    Z_i = 0
    Z_ij = 0
    Z_i_deriv = 0
    Z_ij_deriv = 0
    avg_its = 5
    for i in range(avg_its):
        chis = np.array(chis)
        neighbour_chis = chis[np.random.randint(population_n, size=(population_n, d))]
        i_s = np.array([Z_i_(chi_neighbours, d, allowed_alife_neighbours, attr_size_graph, attractor_graph,
                             homogenous_point_attractor, require_full_rattling, all_rattlers) for chi_neighbours in
                        neighbour_chis])

        ij_s = np.array([observables.calc_marginals_pop(chi1, chi2)['Z_ij'] for chi1, chi2 in
                         chis[np.random.randint(population_n, size=(population_n, 2))]])

        Z_i += np.power(i_s, m_parisi).mean()
        Z_i_deriv += np.where(i_s != 0, np.power(i_s, m_parisi) * np.log(i_s), 0.0).mean()

        Z_ij += np.power(ij_s, m_parisi).mean()
        Z_ij_deriv += np.where(ij_s != 0, np.power(ij_s, m_parisi) * np.log(ij_s), 0.0).mean()
    Z_i /= avg_its
    Z_ij /= avg_its
    Z_i_deriv /= avg_its
    Z_ij_deriv /= avg_its
    s = Z_i_deriv / Z_i - d / 2 * Z_ij_deriv / Z_ij
    Phi_entropy = np.log(Z_i) - d / 2 * np.log(Z_ij)
    complexity = Phi_entropy - m_parisi * s
    return Phi_entropy, complexity, s