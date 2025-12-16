import numpy as np

"""
Functions for initializing the fixed point in a smart way.
"""

def init_balanced(shape):
    chi = np.ones(shape)
    return chi


def init_random_Gauss(shape):
    chi = np.abs(np.random.uniform(size=shape))
    return chi


def init_proportional(shape, init_mag):
    chi = np.ones(shape)
    PROBS = [1 - init_mag, init_mag]
    attr_size_graph = int(len(shape) / 2)

    def prob_next(center_now, parent_prev):
        if center_now == 0:
            if parent_prev == 0:
                return PROBS[0]
            else:
                return PROBS[0] ** 2
        if center_now == 1:
            if parent_prev == 1:
                return PROBS[1]
            else:
                return PROBS[1] ** 2

    for x_center in product([DEAD, ALIFE], repeat=attr_size_graph):
        for x_parent in product([DEAD, ALIFE], repeat=attr_size_graph):
            nodes = [0] * (attr_size_graph * 2)
            # current_state =
            nodes[::2] = x_center
            nodes[1::2] = x_parent
            p = (PROBS[x_parent[0]]
                 * PROBS[x_center[0]]
                 * prob_next(x_center[1], x_parent[0])
                 * prob_next(x_parent[1], x_center[0]))

            chi[tuple(nodes)] = p

    return chi

def init_fixed(shape,chi):
    return chi