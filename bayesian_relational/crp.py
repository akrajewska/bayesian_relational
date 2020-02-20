import numpy as np
from numpy.random import choice, uniform
from scipy.special import gamma, factorial


#graph adjacency matrix

def reassign_nodes(z, i, cls_sizes, K, alpha=1):
    z_prop = z.copy()
    if np.random.random() < (alpha / (alpha + K)):
        m = choice(K)
        z_prop[i] = m
        map(lambda x: x + 1, [cls for cls in z if cls >= m])
    else:
        p = (cls_sizes/len(z))
        z_prop[i] = choice(len(cls_sizes), p=p)
    return z_prop

def update_edges(G, z, i):
    m = z[i]
    for j in range(len(z)):
        if G[i, j] == 0:
            continue
        elif z[j] == m:
            G[j, i] = 0
        elif z[j] > m:
            G[j, i] = -1
        else:
            G[j, i] = 1



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
    G_prop = G.copy()
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
def score(G, z, cls_sizes, alpha=1, beta_parameters=1):
    d = len(z)
    scalar = 1 / factorial(d) * gamma(alpha) / gamma(alpha + d)
    a = np.prod(factorial(cls_sizes))
    edges_counter = {}
    for i in range(d):
        for j in range(d):
            if z[i] > z[j]:
                a, b = z[i], z[j]
                if a == b: continue
                if (a,b) not in edges_counter:
                    edges_counter[a, b] = [0, 0]
                if G[i, j]:
                    edges_counter[a,b][0] += 1
                else:
                    edges_counter[a,b][1] += 1
    b = np.prod([(beta_parameters + edges[0], beta_parameters + edges[1]) for edges in edges_counter])
    _score = scalar * a * b
    return _score


def accept(current_score, last_score):
    return min(1, current_score/last_score)

def step(G, z, scores, alpha=1, beta=1):
    i = choice(len(z))
    cls_sizes = np.bincount(z)
    K = len(cls_sizes)
    z_prop = reassign_nodes(z, i, cls_sizes, alpha)
    G_prop = graph_step(G, K, z, cls_sizes, i)
    current_score = score(G, z, alpha, beta)
    r = uniform()
    if r < current_score:
        scores.append(current_score)
        return G_prop, z_prop
    return G, z

#a > b
# ajacency matrix

def run(G_init, z_init, N, alpha=1, beta=1):
    score_init = score(G_init, z_init, np.bincount(z_init), alpha, beta)
    scores = [score_init]
    G, z = G_init, z_init
    for epoch in range(N):
        G, z = step(G, z, alpha)
    print(scores)


if __name__ == '__main__':
    i = 1
    z = np.array([1,2,2,3])
    cls_sizes = np.array([1,2,1])
    K = 2
    out = reassign_nodes(z, i, cls_sizes, K)
