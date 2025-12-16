
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from bdcm.rules import TotalisiticRule, DEAD, ALIFE
from bdcm.initialization import init_random_Gauss
from bdcm.observables import Observables


def Z_i_(chi, chi_neighbours, d, allowed_alife_neighbours, attr_size_graph, attractor_graph,homogenous_point_attractor,require_full_rattling,all_rattlers):
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
            idx[OUT::2] = c
            idx[IN::2] = x
            idx = tuple(idx)

            p_idx = [0] * (attr_size_graph * 2)
            p_idx[:attr_size_graph] = x
            p_idx[attr_size_graph:] = c
            p_idx = tuple(p_idx)
            prob_alive[p_idx] += chi_neighbours[0][idx]
    for k in range(1, d):
        prob_alive__ = np.zeros_like(prob_alive)
        for c in product([DEAD, ALIFE], repeat=attr_size_graph):
            for x in product([DEAD, ALIFE], repeat=attr_size_graph):
                idx = [0] * attr_size_graph * 2
                idx[OUT::2] = c
                idx[IN::2] = x
                idx = tuple(idx)

                p_idx = [0] * (attr_size_graph * 2)
                p_idx[:attr_size_graph] = x
                p_idx[attr_size_graph:] = [slice(c[p], k + 1 + c[p]) for p in range(attr_size_graph)]
                p_idx = tuple(p_idx)

                o_idx = [0] * (attr_size_graph * 2)
                o_idx[:attr_size_graph] = x
                o_idx[attr_size_graph:] = [slice(None, k + 1) for p in range(attr_size_graph)]
                o_idx = tuple(o_idx)

                prob_alive__[p_idx] += prob_alive[o_idx] * chi_neighbours[k][idx]
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


def check_Nx_allowance(NODES, current_step, next_step, func):
    current = NODES[2 * current_step:2 * current_step + 2]
    next = NODES[2 * next_step:2 * next_step + 2]

    return func(current[OUT], current[IN], next[OUT])

def get_best_mu_field(chi_hat, rho, mu0, m, d, attr_size_graph, last_mu=None):
    def f(mu):
        return (m - get_M(mu, rho,mu0, chi_hat, d, attr_size_graph)) ** 2

    best = fmin(f, 0.0 if last_mu is None else last_mu, disp=False, ftol=0.0000001, xtol=0.0000001)[0]
    return best
from tqdm import tqdm
# raise Exception

OUT = 0
IN = 1


def rs_calculation(its, game, d, chi_comp, c=1, p=0,
                    init_func=init_random_Gauss, alpha=0.99,
                        m_target=None, attraction_factor=2,fix_observable=None, balance_colors=False,
                        log_dir=None,log_suffix='',all_rattlers=None,homogenous_point_attractor=None,require_full_rattling=False,epsilon=None,**kwargs ):

    if homogenous_point_attractor is not None:
        assert homogenous_point_attractor in [0,1]
        assert c == 0

    if all_rattlers is not None:
        assert c == 2
        c = 1

    if epsilon is None:
        # stick to iterations and do not break on convergence
        epsilon = 10e-20

    # preprocess configuration
    has_cycle = c
    c = 0 if c is None else c
    attr_size_graph = c + p
    if fix_observable is not None:
        target_observable, target_value = fix_observable

    config = {**locals(), **kwargs}
    del config['kwargs']

    ALL_ALLOWED = np.arange(d)

    config['init_func'] = config['init_func'].__name__

    game = TotalisiticRule(game,d)
    allowed_alife_neighbours = game.allowed_alife_neighbours
    observables = Observables(**config)
    attractor_graph = [(p +1) % attr_size_graph for p in range(attr_size_graph)]
    attractor_graph[-1] = p if has_cycle is not None else None # also allow to not have cycles

    shape = tuple([2] * (attr_size_graph * 2))
    population_n = 600
    chis = [init_func(shape) for _ in range(population_n)]
    chis = [ chi / chi.sum() for chi in chis]

    init_chi = None

    # create function helper that is chached
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

    # direction: x->y
    ss = []
    i = 0


    converged = False
    dist = []

    while i < its and not converged:

        chis = np.array(chis)
        kk = chis.reshape(population_n, -1)
        dist.append(np.abs(kk - chi_comp.flatten()).mean())
        if i % 10 == 0:
            for k in range(kk.shape[1]):
                plt.hist(kk[:,k],histtype='step')
            #plt.title(chis.mean(axis=0).flatten())
            plt.title(f'{i=}, p={p}')
            plt.xlim(0,0.5)
            plt.show()
            plt.plot(dist)
            plt.axhline(0.0)
            plt.show()
        neighbour_chis = chis[np.random.randint(population_n,size=(population_n,d-1))]
        alp=0.1
        update_mask = np.random.choice([True,False],p=[alp,1-alp],size=(population_n))

        chis = [step(ALL_ALLOWED, all_rattlers, allowed_alife_neighbours, attr_size_graph, attractor_graph,
                     balance_colors, chi,neighbour_chis, d, determine_allowals, homogenous_point_attractor, observables,
                     require_full_rattling) if update else chi for update, chi, neighbour_chis in zip(update_mask, chis,neighbour_chis)]
        i += 1
        """
        chi_old = chi.copy()
        chi = alpha * chi + (1 - alpha) * chi__
        dist = abs(chi_old - chi__).sum()
        if i > 10 and dist  < epsilon:
            print(f'its={i}')
            converged=True
            break
        else:
            #print(dist)
            pass

        """


    #print('convergence',abs(chi_old - chi__).sum())


    print("... done fixed point iteration.")
    log_Z_is = []
    log_Z_ijs = []
    for i in range(10):
        chis = np.array(chis)
        neighbour_chis = chis[np.random.randint(population_n, size=(population_n, d))]
        Zis = [Z_i_(chi,chi_neighbours, d, allowed_alife_neighbours, attr_size_graph, attractor_graph,homogenous_point_attractor,require_full_rattling,all_rattlers) for chi, chi_neighbours in
        zip(chis,neighbour_chis)]
        log_Z_is += [np.log(Z_i) for Z_i in Zis]
        observeds = [observables.calc_marginals_pop(XX[0],XX[1]) for XX in chis[np.random.randint(population_n, size=(population_n, d))]]
        log_Z_ijs += [np.log(observed['Z_ij'] ) for observed in observeds]

        neighbour_chis = chis[np.random.randint(population_n, size=(population_n, d - 1))]
        alp = 0.2
        update_mask = np.random.choice([True, False], p=[alp, 1 - alp], size=(population_n))

        chis = [step(ALL_ALLOWED, all_rattlers, allowed_alife_neighbours, attr_size_graph, attractor_graph,
                     balance_colors, chi, neighbour_chis, d, determine_allowals, homogenous_point_attractor,
                     observables,
                     require_full_rattling) if update else chi for update, chi, neighbour_chis in
                zip(update_mask, chis, neighbour_chis)]




    Phi_RS = np.mean(log_Z_is) - d / 2 * np.mean(log_Z_ijs)

    entropy = Phi_RS #+ observed['Legendre']

    print(entropy)
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


def step(ALL_ALLOWED, all_rattlers, allowed_alife_neighbours, attr_size_graph, attractor_graph, balance_colors, chi, neighbour_chis, d,
         determine_allowals, homogenous_point_attractor, observables, require_full_rattling):
    shape = tuple([2] * (attr_size_graph * 2))
    chi_dp = np.zeros(shape)
    shape = [0] * (attr_size_graph * 2)
    shape[:attr_size_graph] = [2] * attr_size_graph  # first part is the state of the center node
    shape[attr_size_graph:] = [
                                  d] * attr_size_graph  # second part is the number of alive neighbours in the neighbourhood
    # the probability is defines as follows:
    # what is the probability of having k_1 alife neighbours in period p_1 and k_2 in p_2 if x is dead in both? prob_alive[DEAD,DEAD, k1,k2]
    prob_alive = np.zeros(shape)
    # STEP Initialize the DP probability calculation
    for X_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):
        for Y_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):
            idx = [0] * attr_size_graph * 2
            idx[OUT::2] = X_NODE
            idx[IN::2] = Y_NODE
            idx = tuple(idx)

            p_idx = [0] * (attr_size_graph * 2)
            p_idx[:attr_size_graph] = Y_NODE
            p_idx[attr_size_graph:] = X_NODE
            p_idx = tuple(p_idx)
            prob_alive[p_idx] += neighbour_chis[0][idx]
    # STEP Do the DP, this does not incorporate any constraints on the cycles
    for k in range(1, d - 1):
        prob_alive__ = np.zeros_like(prob_alive)
        for X_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):
            for Y_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):
                idx = [0] * attr_size_graph * 2
                idx[OUT::2] = X_NODE
                idx[IN::2] = Y_NODE
                idx = tuple(idx)

                p_idx = [0] * (attr_size_graph * 2)
                p_idx[:attr_size_graph] = Y_NODE
                p_idx[attr_size_graph:] = [slice(X_NODE[p], k + 1 + X_NODE[p]) for p in range(attr_size_graph)]
                p_idx = tuple(p_idx)

                o_idx = [0] * (attr_size_graph * 2)
                o_idx[:attr_size_graph] = Y_NODE

                o_idx[attr_size_graph:] = [slice(None, k + 1) for p in range(attr_size_graph)]
                o_idx = tuple(o_idx)

                prob_alive__[p_idx] += prob_alive[o_idx] * neighbour_chis[k][idx]
        prob_alive = prob_alive__
    # Step to sum only over the allowed combinations for each chi, this changes depending on whether we care about cycles, or cycles with paths as attractors
    for NODES in product([DEAD, ALIFE], repeat=2 * attr_size_graph):
        allowals = []
        Y_NODE = NODES[IN::2]
        X_NODE = NODES[OUT::2]

        if homogenous_point_attractor is not None:
            for curr, nex in enumerate(attractor_graph[:-1]):
                allowed, change_occured = check_Nx_allowance(NODES, curr, nex, determine_allowals)
                allowals.append(allowed)
            current_step = len(attractor_graph) - 1
            # for the last state, check that it goes into [1,1]
            current = NODES[2 * current_step:2 * current_step + 2]
            next = [homogenous_point_attractor, homogenous_point_attractor]

            allowed = allowed_alife_neighbours[current[OUT]] - int(current[IN] == ALIFE)
            allowed = allowed[allowed <= d - 1]
            allowed = allowed[allowed >= 0]

            # if change occurs, we need to change the neighbour
            change_occurs = current[OUT] != next[OUT]
            if change_occurs:
                # we actually want the node to change their color, so we need to make them unahppy in this step!
                allowed = not_(allowed, d)
            allowals.append(allowed)
        elif all_rattlers is not None:
            for curr, nex in enumerate(attractor_graph[:-1]):
                allowed, change_occured = check_Nx_allowance(NODES, curr, nex, determine_allowals)
                allowals.append(allowed)
            current_step = len(attractor_graph) - 1

            opp = lambda x: 0 if x == 1 else 1
            # for the last state, check that it goes into the opposite
            current = NODES[2 * current_step:2 * current_step + 2]
            next = [opp(current[0]), opp(current[1])]

            allowed = allowed_alife_neighbours[current[OUT]] - int(current[IN] == ALIFE)
            allowed = allowed[allowed <= d - 1]
            allowed = allowed[allowed >= 0]

            # if change occurs, we need to change the neighbour
            change_occurs = current[OUT] != next[OUT]
            if change_occurs:
                # we actually want the node to change their color, so we need to make them unahppy in this step!
                allowed = not_(allowed, d)
            allowals.append(allowed)

        else:
            for curr, nex in enumerate(attractor_graph):
                if nex is None:
                    allowals.append(ALL_ALLOWED)
                    continue

                allowed, change_occured = check_Nx_allowance(NODES, curr, nex, determine_allowals)
                if curr == len(attractor_graph) - 1 and require_full_rattling and not change_occured:
                    allowals.append([])  # all are allowed where a change occured?
                else:
                    allowals.append(allowed)

        # c[::2] will be the center node over all attr_size_graph
        chi_dp[tuple(NODES)] = prob_alive[tuple(X_NODE)][np.ix_(*allowals)].sum()
    # STEP to append the priors to the distribution
    for x_center in product([DEAD, ALIFE], repeat=attr_size_graph):
        for x_parent in product([DEAD, ALIFE], repeat=attr_size_graph):
            X_NODE = [0] * (attr_size_graph * 2)
            X_NODE[::2] = x_center
            X_NODE[1::2] = x_parent
            chi_dp[tuple(X_NODE)] *= observables.g(x_center, x_parent).prod()
    if balance_colors:
        for NODES in product([DEAD, ALIFE], repeat=2 * attr_size_graph):
            # set the inputs such that the inverse of every item is equal to its color symmetric item
            SYM_NODES = tuple(0 if kk == 1 else 1 for kk in NODES)
            temp = (chi_dp[tuple(NODES)] + chi_dp[SYM_NODES]) / 2
            chi_dp[tuple(NODES)] = temp
            chi_dp[tuple(SYM_NODES)] = temp
    chi__ = chi_dp.copy()
    chi__ = chi__ / chi__.sum()
    return chi__
