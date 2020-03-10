import numpy as np
from numpy.random import choice, uniform
from scipy.special import gamma, factorial

from data.initial_data import generate_solution_classes, generate_initial_graph

# graph adjacency matrix

Z = generate_solution_classes(7)


def reassign_nodes(z, i, cls_sizes, alpha=1):
    z_prop = z.copy()
    N = len(z)
    u = np.random.random()
    if u < (alpha / (alpha + N - 1)):
        z_tmp = np.delete(z, i)
        cls_sizes = np.bincount(z_tmp)
        K = len(cls_sizes)
        m = choice(K + 1)
        z_new = list(map(lambda x: x + 1 if x >= m else x, z_prop))
        z_new[i] = m
    else:
        p = (cls_sizes / len(z))
        z_prop[i] = choice(len(cls_sizes), p=p)
        z_new = z_prop
    if 0 not in z_new:
        z = [x - 1 for x in z_new]
    return z

def update_edges(G, z, i):
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



def graph_step(G, K, z, cls_sizes, i):
    G_prop = G.copy()
    update_edges(G_prop, z, i)
    # choose to classes randomly
    a, b = choice(K, size=2, replace=False)
    node_indexes = []
    for cls in [a, b]:
        indices = np.argwhere(z == cls)
        # choose element from class
        j = choice(len(indices))
        node_indexes.append(indices[j])

    r = choice(1)
    if r:
        return G
    elif a > b:
        G_prop[node_indexes[0], node_indexes[1]] = 1
        G_prop[node_indexes[1], node_indexes[0]] = 0
    else:
        G_prop[node_indexes[0], node_indexes[1]] = 0
        G_prop[node_indexes[1], node_indexes[0]] = 1
    return G_prop


#TODO jak to jest z parametrami beta
def score(G, z, alpha=1, beta_parameters=1):
    cls_sizes = np.bincount(z)
    # number of classes
    K = len(cls_sizes)
    d = G.shape[1]
    scalar = 1 / factorial(d) * gamma(alpha) / gamma(alpha + d) * alpha ** K
    A = np.prod(factorial(cls_sizes))

    edges_counter = {}
    for i in range(d):
        for j in range(i):
            if z[i] > z[j]:
                a, b = z[i], z[j]
                if (a, b) not in edges_counter:
                    edges_counter[a, b] = [0, 0]
                if G[i, j]:
                    edges_counter[a, b][0] += 1
                else:
                    edges_counter[a, b][1] += 1
    B = np.prod([(beta_parameters + edges_counter[edges][0], beta_parameters + edges_counter[edges][1]) for edges in
                 edges_counter])
    _score = scalar * A * B
    return _score


def accept(current_score, last_score):
    return min(1, current_score/last_score)

def step(G, z, scores, alpha=1, beta=1):
    i = choice(len(z))
    cls_sizes = np.bincount(z)
    K = len(cls_sizes)
    z_prop = reassign_nodes(z, i, cls_sizes, alpha)
    G_prop = graph_step(G, K, z, cls_sizes, i)
    current_score = score(G, z_prop)
    r = uniform()
    if r < 0.5:
        scores.append(current_score)
        return G_prop, z_prop, scores
    return G, z, scores


# a > b
# ajacency matrix

def run(G_init, z_init, N, alpha=1, beta=1):
    score_init = score(G_init, z_init, alpha, beta)
    scores = [score_init]
    G, z = G_init, z_init
    for epoch in range(N):
        G, z, scores = step(G, z, scores, alpha)
    return G, z, scores


def recall(output, solution):
    pass


if __name__ == '__main__':
    G = np.zeros((6, 6))
    z = generate_solution_classes(2)
    print(score(G, z))
    G = generate_initial_graph(1)
    print(score(G, z))
