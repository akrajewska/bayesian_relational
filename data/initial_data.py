import numpy as np


def generate_initial_classes(phase_number):
    z = np.array([0, 0, 1])
    for i in range(phase_number):
        if i % 2 == 0:
            w = np.array([1, 1, 0])
        else:
            w = np.array([0, 0, 1])
        z = np.concatenate([z, w])
    return z


def observation_to_graph(G, phase_number):
    z = generate_initial_classes(phase_number)
    for i in range(3 * phase_number):
        for j in range(3 * phase_number):
            if z[i] < z[j]:
                G[i, j] = 1


def generate_initial_graph(phase_number):
    d = 3 * (phase_number + 1)
    G = np.zeros((d, d))
    observation_to_graph(G, phase_number)
    for i in range(3 * phase_number + 1, 3 * phase_number + 3):
        for j in range(d):
            G[i, j] = 1
    return G


if __name__ == '__main__':
    generate_initial_graph(1)
