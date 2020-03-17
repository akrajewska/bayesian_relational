from itertools import combinations

import numpy as np
from numpy import random
from scipy.special import gamma, factorial


def remove_node(z: np.ndarray, i: int, z_bincount: np.ndarray = None):
    """
    Remove i-th node and update class labels to be in sequence
    """
    if z_bincount is None:
        z_bincount = np.bincount(z)
    np.delete(z, i, axis=0)
    _cls = z[i]
    # if i-th node was in class of its own, relabel classes
    if z_bincount[_cls] == 1:
        for label in z:
            if label > z[i]:
                label -= 1


def reassign_nodes(z: np.ndarray, i: int, alpha: int = 1):
    """
    Reassing i-node according to CRP
    """
    z_bincount = np.bincount(z)
    remove_node(z, i, z_bincount)
    u = np.random.random()
    # number of classes
    N = len(z)
    z_bincount = np.bincount(z)
    # number of classes
    K = len(z_bincount)
    if u < (alpha / (alpha + N - 1)):
        m = random.choice(K + 1)
        z_new = list(map(lambda x: x + 1 if x >= m else x, z))
        z_new[-1] = m
    else:
        p = (z_bincount / len(z))
        z[-1] = random.choice(len(z_bincount), p=p)
    return z


def update_edges(G: np.ndarray, z: np.ndarray, i: int):
    m = z[i]
    for j in range(len(z)):
        if z[j] == m:
            G[j, i] = 0
            G[i, j] = 0
        elif z[j] > m:
            G[j, i] = 0
            G[i, j] = 1
        else:
            G[j, i] = 1
            G[i, j] = 0


def graph_step(G: np.ndarray, z: np.ndarray, i: int):
    z_bincount = np.bincount(z)
    K = len(z_bincount)
    G_prop = G.copy()
    update_edges(G_prop, z, i)
    # choose to classes randomly
    a, b = random.choice(K, size=2, replace=False)
    node_indexes = []
    for cls in [a, b]:
        indices = np.argwhere(z == cls)
        # choose element from class
        j = random.choice(len(indices))
        node_indexes.append(indices[j])

    r = random.choice(1)
    if r:
        return G
    elif a > b:
        G_prop[node_indexes[0], node_indexes[1]] = 1
        G_prop[node_indexes[1], node_indexes[0]] = 0
    else:
        G_prop[node_indexes[0], node_indexes[1]] = 0
        G_prop[node_indexes[1], node_indexes[0]] = 1
    return G_prop


def count_edges(G: np.ndarray, z: np.ndarray, z_bincount: np.ndarray = None, K: int = None):
    """
    For each pair of classes counts number of present and absent edges of G
    :return:
    """
    if z_bincount is None:
        z_bincount = np.bincount(z)
    if K is None:
        K = len(z_bincount)
    # dict with keyes: ordered pair of classes from z vector
    present_edges = {pair: 0 for pair in combinations(range(K), 2)}
    absent_edges = {pair: 0 for pair in combinations(range(K), 2)}
    for i in range(G.shape[0]):
        a = z[i]
        for j in range(G.shape[1]):
            b = z[j]
            if a == b: continue
            if a > b:
                if G[i, j]:
                    present_edges[(b, a)] += 1
                else:
                    absent_edges[(b, a)] += 1
            else:
                if G[j, i]:
                    present_edges[(a, b)] += 1
                else:
                    absent_edges[(a, b)] += 1
    return np.fromiter(present_edges.values(), dtype=float), np.fromiter(absent_edges.values(), dtype=float)


def conditional_probability(G: np.ndarray, z: np.ndarray, beta: float, z_bincount: np.ndarray = None, K: int = None):
    """
    Probability of graph G given z classification
    """
    if z_bincount is None:
        z_bincount = np.bincount(z)
    if K is None:
        K = len(z_bincount)
    present_edges, absent_edges = count_edges(G, z, z_bincount, K)
    p_G_given_z = np.prod([(beta + present_edges[i], beta + present_edges[i]) for i in range(len(present_edges))])
    return p_G_given_z


def z_nodes_permutions(z_bincount: np.ndarray):
    """
    :param z_bincount: vector counting occurrences of each class
    :return: number of nodes permutations underwhich z can be obtained
    """
    return np.prod(factorial(z_bincount))


# TODO jak to jest z parametrami beta
def score(G: np.ndarray, z: np.ndarray, alpha=1, beta=1):
    """
    Score function is given by a joint probability of pair (G, z)
    :param G: adjacency matrix of graph structure
    :param z: categorization vector
    :param alpha: CRP parameter
    :param beta: beta distritbution parameters
    :return: score value: float
    """
    z_bincount = np.bincount(z)
    # number of classes
    K = len(z_bincount)
    d = G.shape[1]
    scalar = 1 / factorial(d) * gamma(alpha) / gamma(alpha + d) * alpha ** K
    A = z_nodes_permutions(z_bincount)
    B = conditional_probability(G, z, beta, z_bincount, K)
    _score = scalar * A * B
    return _score


def accept(current_score, last_score):
    return min(1, current_score / last_score)


def step(G: np.ndarray, z: np.ndarray, scores: list, alpha: float = 1, beta: float = 1):
    i = random.choice(len(z))
    z_prop = reassign_nodes(z, i, alpha)
    G_prop = graph_step(G, z, i)
    current_score = score(G, z_prop)
    r = random.uniform()
    if r < 0.5:
        scores.append(current_score)
        return G_prop, z_prop, scores
    return G, z, scores


def run(G_init: np.ndarray, z_init: np.ndarray, N: int = 1000, alpha=1, beta=1):
    """
    :param G_init: adjacency matrix of initial graph structure
    :param z_init: initial categorization vector
    :param N: number of epochs
    :param alpha: CRP parameter
    :param beta: beta distribution parameters
    :return: (G, z, scores) output graph, catagerization vector and list of scores
    """
    score_init = score(G_init, z_init, alpha, beta)
    scores = [score_init]
    G, z = G_init, z_init
    for epoch in range(N):
        G, z, scores = step(G, z, scores, alpha)
    return G, z, scores
