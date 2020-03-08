import numpy as np


def generate_solution_classes(phase_number):
    z = np.array([0, 0, 1])
    for i in range(phase_number):
        if i % 2 == 0:
            w = np.array([1, 1, 0])
        else:
            w = np.array([0, 0, 1])
        z = np.concatenate([z, w])
    return z


def observation_to_graph(G, phase_number):
    z = generate_solution_classes(phase_number)
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


def solution(phase_number):
    d = 3 * (phase_number + 1)
    G = np.zeros((d, d), dtype=bool)
    z = generate_solution_classes(phase_number)
    for i in range(3 * phase_number + 1):
        for j in range(3 * phase_number + 1):
            if z[i] < z[j]:
                G[i, j] = 1
    return G


def interaction_update(G, z, phase_number):
    probe_index = 3 * phase_number
    target_index = probe_index - 1
    if z[probe_index] < z[target_index]:
        G[probe_index, target_index] = 1
        G[target_index, probe_index] = 0
    else:
        G[probe_index, target_index] = 0
        G[target_index, probe_index] = 1


if __name__ == '__main__':
    z = generate_solution_classes(1)
