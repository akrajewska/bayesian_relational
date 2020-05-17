from itertools import combinations

import numpy as np
from numpy import random
from scipy.special import gamma, factorial

from data.initial_data import solution, correct_classes, interaction_update


SOLUTION, z_solution = solution(100)
PHASE_NUMBER =7

def remove_node(z: np.ndarray, i: int, z_bincount: np.ndarray = None):
    """
    Remove i-th node and update class labels to be in sequence
    """
    if z_bincount is None:
        z_bincount = np.bincount(z)

    _cls = z[i]
    z = np.delete(z, i, axis=0)
    # if i-th node was in class of its own, relabel classes
    if z_bincount[_cls] == 1:
        relabeled_z = [z[i]-1 if z[i] > _cls else z[i] for i in range(len(z))]
        z = relabeled_z
    return z


def reassign_nodes(z: np.ndarray, i: int, alpha: int = 1):
    """
    Reassing i-node according to CRP
    """
    z_bincount = np.bincount(z)
    z = remove_node(z, i, z_bincount)
    u = np.random.random()
    # number of classes
    N = len(z)
    #
    # z_to_sample = [x for x in z if x != m ]
    z_bincount = np.bincount(z)
    if True:
        # number of classes
        K = len(z_bincount)
        if u < (alpha / (alpha + N - 1)):
            m = random.choice(K + 1)
            z_new = list(map(lambda x: x + 1 if x >= m else x, z))
            z = np.insert(z_new, i, m)
        else:
            p = (z_bincount / len(z))
            m = random.choice(len(z_bincount), p=p)
            z = np.insert(z, i, m)
    return z


def update_edges(G: np.ndarray, z: np.ndarray, i: int):
    for i in range(len(z)):
        for j in range(len(z)):
            if z[i] < z[j]:
                #contradicts
                if G[j, i] == 1:
                    G[j, i] = 0
                    G[i, j] = 1
            elif z[i] > z[j]:
                if G[i, j] == 1:
                    G[i, j] = 0
                    G[j, i] = 1
            else:
                G[j, i] = 0
                G[i, j] = 0



def contradict(G: np.ndarray, z: np.ndarray):
    for i in range(len(z)):
        for j in range(len(z)):
            if z[i] <= z[j]:
                #contradicts
                if G[j, i] == 1:
                    return True
    return False

def _sample_one(G: np.ndarray, i: int, j: int):
    r = random.choice(1)
    if r:
        G[i, j] = 1

    else:
        G[i, j] = 0



def _sample_from_all(G: np.ndarray, z: np.ndarray):
    z_bincount = np.bincount(z)
    K = len(z_bincount)
    _a, _b = random.choice(K, size=2, replace=False)

    a, b = (_a, _b) if _a < _b else (_b, _a)

    node_indexes = []
    for cls in [a, b]:
        indices = np.argwhere(z == cls)
        # choose element from class
        j = random.choice(len(indices))
        node_indexes.append(indices[j][0])
    i, j = node_indexes
    _sample_one(G, i, j)

def _sample_from_new(G: np.ndarray, z: np.ndarray):
    new_node_idx = len(z) - random.choice([1, 2, 3])
    _a = z[new_node_idx]
    z_bincount = np.bincount(z)

    z_without_a_bincount = np.delete(z_bincount, _a)
    K = len(z_without_a_bincount)

    _b = random.choice(K)
    indices = np.argwhere(z == _b)
    j = random.choice(len(indices))
    j = indices[j][0]
    i = new_node_idx
    _sample_one(G, i, j)


def _sample_edges(G: np.ndarray, z: np.ndarray, beta:tuple):
    new_node_idx = len(z) - random.choice([1, 2, 3])
    _a = z[new_node_idx]
    z_bincount = np.delete(np.bincount(z), _a)
    _b = random.choice(z_bincount)
    _b_indices = np.argwhere(z == _b)
    for idx in _b_indices:
        r = random.uniform()
        _beta = random.beta(beta[0], beta[1])
        if r<_beta:
            if _a < _b:
                G[new_node_idx, idx] = 1
                G[idx, new_node_idx] = 0
            else:
                G[idx, new_node_idx] = 1
                G[new_node_idx, idx] = 0
        else:
            if _a < _b:
                G[new_node_idx, idx] = 0
                G[idx, new_node_idx] = 0
            else:
                G[idx, new_node_idx] = 0
                G[new_node_idx, idx] = 0


def get_node_indices(z: np.ndarray):
    return _sample_from_all(z)

def sample_graph(G: np.ndarray, z: np.ndarray, beta: tuple):
    # _sample_from_all(G, z)
    _sample_edges(G, z, beta)

def graph_step(G: np.ndarray, z: np.ndarray, i: int, after_interaction: bool=False, beta: tuple=(0.5, 0.5)):
    G_prop = G.copy()
    update_edges(G_prop, z, i)
    # draw(G_prop, z, "after_update")

    sample_graph(G, z, beta)
    # draw(G_prop, z, "after sample")
    for j in range(len(z)-3):
        for l in range(len(z)-3):
            G_prop[j,l] = SOLUTION[j,l]
    # draw(G_prop, z, "after adjus")
    if after_interaction:
        interaction_update(G_prop, SOLUTION[2], phase_number=PHASE_NUMBER)

    return G_prop


def count_edges(G: np.ndarray, z: np.ndarray, z_bincount: np.ndarray = None, K: int = None):
    """
    For each pair of classes counts number of present and absent edges of G
    :return: tuple
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
            if G[i, j]:
                if a < b:
                    present_edges[(a, b)] += 1
            else:
                if a < b:
                    absent_edges[(a, b)] += 1

    return np.fromiter(present_edges.values(), dtype=float), np.fromiter(absent_edges.values(), dtype=float)


def conditional_probability(G: np.ndarray, z: np.ndarray, beta: tuple, z_bincount: np.ndarray = None, K: int = None):
    """
    Probability of graph G given z classification
    """
    if z_bincount is None:
        z_bincount = np.bincount(z)
    if K is None:
        K = len(z_bincount)
    present_edges, absent_edges = count_edges(G, z, z_bincount, K)
    beta1, beta2 = beta
    p_G_given_z = np.prod([(beta1 + present_edges[i], beta2 + present_edges[i]) for i in range(len(present_edges))])
    return p_G_given_z


def z_nodes_permutions(z_bincount: np.ndarray):
    """
    :param z_bincount: vector counting occurrences of each class
    :return: number of nodes permutations underwhich z can be obtained
    """
    return np.prod(factorial(z_bincount))



# # TODO jak to jest z parametrami beta
def score(G: np.ndarray, z: np.ndarray, alpha=1, beta=(0.5, 0.5)):
    """
    Score function is given by a joint probability of pair (G, z)
    :param G: adjacency matrix of graph structure
    :param z: categorization vector
    :param alpha: CRP parameter
    :param beta: beta distritbution parameters
    :return: score value: float
    """
    if contradict(G,z):
        return 0

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



def admissible(z: np.ndarray):
    n = len(z) - 3

    observations = SOLUTION[:n, :n]

    if contradict(observations, z[:n]):
        return False

    return True



def step(G: np.ndarray, z: np.ndarray, scores: list, max_score: float, G_max: np.ndarray, z_max: np.ndarray,after_interaction: bool=False,alpha: float = 0.5, beta: tuple = (0.5, 0.5)):
    N = len(z)
    # i = random.choice(range(N))
    i = random.choice([N-3, N-2, N-1])
    z_prop = reassign_nodes(z, i, alpha)
    if admissible(z_prop):
        G_prop = graph_step(G, z_prop, i, after_interaction=after_interaction, beta=beta)

        current_score = score(G_prop, z_prop)
        # draw(G_prop, z_prop, f'current score {current_score}')
        if current_score == 0:
            return G, z, scores, max_score, G_max, z_max
        r = random.uniform()
        last_score = scores[-1]
        alfa = accept(current_score, last_score)
        if r < alfa:
            if current_score > max_score:
                max_score = current_score
                G_max, z_max = G_prop, z_prop
            scores.append(current_score)
            return G_prop, z_prop, scores, max_score, G_max, z_max
    return G, z, scores, max_score, G_max, z_max


def run(G_init: np.ndarray, z_init: np.ndarray, max_score: int=None, after_interaction: bool=False, N: int = 100, alpha: float =1, beta: tuple = (1, 1)):
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
    if not max_score:
        max_score = score_init
    G, z = G_init, z_init
    G_max, z_max = G_init, z_init
    for epoch in range(N):
        G, z, scores, max_score, G_max, z_max = step(G, z, scores, max_score, G_max, z_max, after_interaction=after_interaction,  alpha=alpha, beta=beta)
    return G, z, scores, max_score, G_max, z_max
