import numpy as np


def correct_classes(phase: int):
    """
    :return: z: np.ndarray
    """
    z = np.array([0, 0, 1])
    for i in range(phase):
        if i % 2 == 0:
            w = np.array([1, 1, 0])
        else:
            w = np.array([0, 0, 1])
        z = np.concatenate([z, w])
    return z


def observation_to_graph(G: np.ndarray, phase: int):
    """
    Generate graph consistent with seen interactions
    """
    z = correct_classes(phase)
    for i in range(3 * phase):
        for j in range(3 * phase):
            if z[i] < z[j]:
                G[i, j] = 1


def generate_initial_graph(phase: int):
    d = 3 * (phase + 1)
    G = np.zeros((d, d))
    observation_to_graph(G, phase)
    for i in range(3 * phase + 1, 3 * phase + 3):
        for j in range(d):
            G[i, j] = 1
    return G


def solution(phase: int):
    """
    :param phase_number:
    :return: correct graph structure G
    """
    d = 3 * (phase + 1)
    G = np.zeros((d, d))
    z = correct_classes(phase)
    for i in range(3 * (phase + 1)):
        for j in range(3 * (phase + 1)):
            a = z[i]
            b = z[j]
            if z[i] < z[j]:
                G[i, j] = 1
    return G


def interaction_update(G: np.ndarray, z: np.ndarray, phase_number: int):
    """
    Update edges with single interaction info
    """
    probe_index = 3 * phase_number
    target_index = probe_index - 1
    if z[probe_index] < z[target_index]:
        G[probe_index, target_index] = 1
        G[target_index, probe_index] = 0
    else:
        G[probe_index, target_index] = 0
        G[target_index, probe_index] = 1


if __name__ == '__main__':
    z = correct_classes(1)
    G = solution(1)
    print("PAWEL")
