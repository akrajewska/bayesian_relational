import numpy as np
from numpy.random import choice

def reassign_nodes(z, i, cls_sizes, K, alpha=1):

    if np.random.random() < (1. * alpha / (alpha + K)):
        m = choice(K)
        z[i] = m
        map(lambda cls: cls+1 for cls in z if cls >= m)
    else:
        p = (cls_sizes/len(z))
        z[i] = choice(len(cls_sizes), p=p)
    return i

def update_edges(G, z, i):
    m = z[i]
    for j in range(z):
        if G[i, j] == 0:
            continue
        elif z[j] == m:
            G[j, i] = 0
        elif z[j] > m:
            G[j, i] = -1
        else:
            G[j, i] = 1


def graph_step(G, K, z, cls_sizes):
    a, b = choice(K, 2)
    in_cls_indexes = {a: choice(cls_sizes[a]), b: choice(cls_sizes[b])}
    node_indexes = {}
    for label in z:
        if label not in (a, b):
            continue
        if in_cls_indexes[a] == 0 and in_cls_indexes[b] == 0:
            break
        if in_cls_indexes[label] == 0:
            node_indexes[label] = z.index(label) - 1
        in_cls_indexes[label] -= 1
    r = choice(1)
    if r:
        return G
    elif a > b:
        G[node_indexes[a], node_indexes[b]] = -1
        G[node_indexes[b], node_indexes[a]] = 1
    else:
        G[node_indexes[a], node_indexes[b]] = 1
        G[node_indexes[b], node_indexes[a]] = -1



def step(G, z, alpha=1):
    i = choice(len(z))
    cls_sizes = np.bincount(z)
    K = len(cls_sizes)
    reassign_nodes(z, i, cls_sizes, alpha)
    update_edges(G, z, i)
    graph_step(G, z, K)


if __name__ == '__main__':
    i = 1
    z = np.array([1,2,2,3])
    cls_sizes = np.array([1,2,1])
    K = 2
    out = reassign_nodes(z, K, cls_sizes)
