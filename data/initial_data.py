import numpy as np
from non_reversible_mcmc.point import Point
from numpy import random

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

def initial_classes(phase: int):
    """
    :return: z: np.ndarray
    """
    z = np.array([0, 0, 1])
    # for i in range(phase - 1):
    #     w = np.array([1, 1, 1])
    #     z = np.concatenate([z, w])
    for i in range(phase-1):
        if i % 2 == 0:
            w = np.array([1, 1, 0])
        else:
            w = np.array([0, 0, 1])
        z = np.concatenate([z, w])
    w = np.array([0,0,0])
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


def initial_graph(phase: int, initial_type: str):
    d = 3 * (phase + 1)
    G = np.zeros((d, d))
    observation_to_graph(G, phase)
    for i in range(3 * phase, 3 * phase + 3):
        for j in range(3 * phase + 1):
            if initial_type == "A":
                G[i, j] = 1
            elif initial_type == "B":
                r = random.choice(2)
                if r:
                    G[i, j] = 1
                else:
                    G[i, j] = 0
            elif initial_type == "C":
                G[i, j] = 0

    # for j in range(d):
    #     G[d-1, j] = 1

    return G

def initial_point(phase: int, initial_type: str):
    G = initial_graph(phase, initial_type)
    z = initial_classes(phase)
    observation = solution(phase)[0]
    z_solution = solution(phase)[1]
    return Point(G, z, observation, z_solution)

def solution(phase: int=0, dim: int=0):
    """
    :param phase_number:
    :return: correct graph structure G
    """
    if not phase and not dim:
        raise ValueError
    d = 3 * (phase + 1)
    G = np.zeros((d, d))
    z = correct_classes(phase)
    for i in range(3 * (phase + 1)):
        for j in range(3 * (phase + 1)):
            a = z[i]
            b = z[j]
            if z[i] < z[j]:
                G[i, j] = 1
    return G, z


def interaction_update(G: np.ndarray, z: np.ndarray, phase_number: int):
    """
    Update edges with single interaction info
    """
    probe_index = -1
    # print(f"Updating edge for probe with index {probe_index}")
    target_index = -4
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
