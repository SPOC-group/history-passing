import numpy as np
import networkx as nx
from collections import defaultdict
import random

def generate_truncated_Poisson(N=100, c=3):
    C=np.zeros(N, dtype='int')
    for i in range(N):
        poisson=np.random.poisson(lam=c)
        while(poisson<2):
            poisson=np.random.poisson(lam=c)
        C[i]=poisson
    return np.sort(C) # We sort the truncated Poisson to simplify the graph generations process

def generate_truncated_Poisson_graph(N=1000,c=2.14913, max_tries=10):
    poissons=generate_truncated_Poisson(N,c)
    num_degree=[np.count_nonzero(poissons==i) for i in range(2, np.max(poissons)+1)]
    indexes_degree_separation=[0]
    for idx,num in enumerate(num_degree):
        indexes_degree_separation.append(np.sum(num_degree[:(idx+1)]))
    G=nx.random_regular_graph(2,N)
    for i in range(len(indexes_degree_separation)-2):
        for node in range(indexes_degree_separation[i+1], indexes_degree_separation[i+2]):
            failed_to_find=False
            if node==N-1:
                if G.degree(node)<poissons[-1]:
                    print("Last node could not have the exact number of degrees")
                break
            while (G.degree(node)<poissons[node]).any() and not failed_to_find:
                nodes_to_choose_from=np.arange(node+1,N)
                for t in range(max_tries):
                    chosen_node=int(rng.choice(nodes_to_choose_from,size=1))
                    if (G.degree(chosen_node)<poissons[chosen_node]).any():
                        if not G.has_edge(node,chosen_node):
                            G.add_edge(node, chosen_node)
                            break
                    if t==max_tries-1:
                        print('could not find a suitable node for node ',node,' after ', max_tries,' tries...')
                        failed_to_find=True

    return G

from collections import defaultdict
import random
# This is a modified version of networkx random_regular_graph that return a set of tupples for the edges. 
# Simply apply np.array(list(random_regular_graph_edges)) to obtain a numpy array containing the edges.
def random_regular_graph_edges(d, n, seed=None):
    r"""Returns a random $d$-regular graph on $n$ nodes.

    The resulting graph has no self-loops or parallel edges.

    Parameters
    ----------
    d : int
      The degree of each node.
    n : integer
      The number of nodes. The value of $n \times d$ must be even.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    The nodes are numbered from $0$ to $n - 1$.

    Kim and Vu's paper [2]_ shows that this algorithm samples in an
    asymptotically uniform way from the space of random graphs when
    $d = O(n^{1 / 3 - \epsilon})$.

    Raises
    ------

    NetworkXError
        If $n \times d$ is odd or $d$ is greater than or equal to $n$.

    References
    ----------
    .. [1] A. Steger and N. Wormald,
       Generating random regular graphs quickly,
       Probability and Computing 8 (1999), 377-396, 1999.
       http://citeseer.ist.psu.edu/steger99generating.html

    .. [2] Jeong Han Kim and Van H. Vu,
       Generating random regular graphs,
       Proceedings of the thirty-fifth ACM symposium on Theory of computing,
       San Diego, CA, USA, pp 213--222, 2003.
       http://portal.acm.org/citation.cfm?id=780542.780576
    """
    if (n * d) % 2 != 0:
        raise ValueError("n * d must be even")

    if not 0 <= d < n:
        raise ValueError("the 0 <= d < n inequality must be satisfied")

    random.seed(seed)
    def _suitable(edges, potential_edges):
        # Helper subroutine to check if there are suitable edges remaining
        # If False, the generation of the graph has failed
        if not potential_edges:
            return True
        for s1 in potential_edges:
            for s2 in potential_edges:
                # Two iterators on the same dictionary are guaranteed
                # to visit it in the same order if there are no
                # intervening modifications.
                if s1 == s2:
                    # Only need to consider s1-s2 pair one time
                    break
                if s1 > s2:
                    s1, s2 = s2, s1
                if (s1, s2) not in edges:
                    return True
        return False

    def _try_creation():
        # Attempt to create an edge set

        edges = set()
        stubs = list(range(n)) * d

        while stubs:
            potential_edges = defaultdict(lambda: 0)
            random.shuffle(stubs)
            stubiter = iter(stubs)
            for s1, s2 in zip(stubiter, stubiter):
                if s1 > s2:
                    s1, s2 = s2, s1
                if s1 != s2 and ((s1, s2) not in edges):
                    edges.add((s1, s2))
                else:
                    potential_edges[s1] += 1
                    potential_edges[s2] += 1

            if not _suitable(edges, potential_edges):
                return None  # failed to find suitable edge set

            stubs = [
                node
                for node, potential in potential_edges.items()
                for _ in range(potential)
            ]
        return edges

    # Even though a suitable edge set exists,
    # the generation of such a set is not guaranteed.
    # Try repeatedly to find one.
    edges = _try_creation()
    while edges is None:
        edges = _try_creation()

    return edges