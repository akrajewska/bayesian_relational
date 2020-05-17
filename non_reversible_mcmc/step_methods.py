import numpy as np
from numpy import random
from .point import Point
from data.initial_data import interaction_update
from copy import deepcopy
# PHASE_NUMBER=7
from draw.draw import draw

from non_reversible_mcmc import logging

class Step:

    reassigned_idx = -1

    def __init__(self, alpha: float, beta: tuple, phase_number: int):
        self.phase_number = phase_number
        self.alpha = alpha

    def _chose_node(self, z: np.ndarray):
        N = len(z)
        self.reassigned_idx = random.choice([N - 3, N - 2, N - 1])
        # self.reassigned_idx = random.choice(range(N))
        m = z[self.reassigned_idx]
        logging.debug(f'Reassigning {self.reassigned_idx} node of class {m}')

    def _remove_node(self, z: np.ndarray, z_bincount: np.ndarray = None):
        """
        Remove i-th node and update class labels to be in sequence
        """
        i = self.reassigned_idx
        if z_bincount is None:
            z_bincount = np.bincount(z)

        _cls = z[i]
        z = np.delete(z, i, axis=0)
        # if i-th node was in class of its own, relabel classes
        if z_bincount[_cls] == 1:
            relabeled_z = [z[i] - 1 if z[i] > _cls else z[i] for i in range(len(z))]
            z = relabeled_z
        return z


    def _categorization_step(self, z: np.ndarray):
        """
        Reassing i-node according to CRP
        """
        i = self.reassigned_idx
        z_bincount = np.bincount(z)
        z = self._remove_node(z=z, z_bincount=z_bincount)
        u = np.random.random()
        # number of classes
        N = len(z)
        #
        # z_to_sample = [x for x in z if x != m ]
        z_bincount = np.bincount(z)
        if True:
            # number of classes
            K = len(z_bincount)
            if u < (self.alpha / (self.alpha + N - 1)):
                m = random.choice(K + 1)
                z_new = list(map(lambda x: x + 1 if x >= m else x, z))
                z = np.insert(z_new, i, m)
            else:
                p = (z_bincount / len(z))
                m = random.choice(len(z_bincount), p=p)
                z = np.insert(z, i, m)
        logging.debug(f'Assigning class {m} to node {i}')
        return z

    def _update_edges(self, G: np.ndarray, z: np.ndarray):
        for i in range(len(z)):
            for j in range(len(z)):
                if z[i] < z[j]:
                    # contradicts
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

    def step(self, point: Point, after_interaction: bool=False):
        new_point = deepcopy(point)
        z = new_point.z
        self._chose_node(z)
        z_prop = self._categorization_step(z)
        if new_point.z_admissible(z_prop):
            new_point.z = z_prop
            G_prop= self._graph_step(p=new_point, after_interaction=after_interaction)
            new_point.G = G_prop

        # point = Point(G_prop, z_prop)
        return new_point

class OneEdgeStep(Step):

    def _sample_one(self, G: np.ndarray, i: int, j: int):
        r = random.choice(2)
        if r:
            logging.debug(f'Adding edge from {i} to {j}')
            G[i, j] = 1
        else:
            logging.debug(f'Removing or not adding edge from {i} to {j}')
            G[i, j] = 0


    def _sample_from_all(self, p: Point):
        G = p.G
        z = p.z
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
        self._sample_one(G, i, j)

    def _sample_from_new(self, p: Point):
        G = p.G
        z = p.z
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
        self._sample_one(G, i, j)

    def _graph_step(self, p: Point, after_interaction: bool=False):
        G_prop = p.G
        self._update_edges(G_prop, p.z)
        self._sample_from_all(p)
        p.observation_update()
        if after_interaction:
            interaction_update(p.G, p.z_solution, phase_number=self.phase_number)
        # draw(p.G, p.z)
        return p.G



class ManyEdgesStep(Step):

    def __init__(self, alpha: float, beta: tuple, phase_number:int):
        super(ManyEdgesStep, self).__init__(alpha=alpha, beta=beta, phase_number=phase_number)
        self.beta = beta

    def _sample_edges(self, p: Point):
        G = p.G
        z = p.z
        new_node_idx = p.dim - random.choice([1, 2, 3])
        _a = z[new_node_idx]
        _b = random.choice(np.delete(np.unique(z), _a))
        _b_indices = np.argwhere(z == _b)
        for idx in _b_indices:
            r = random.uniform()
            _beta = random.beta(*self.beta)
            if r < _beta:
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
        p.G = G


    def _graph_step(self, p: Point, after_interaction: bool=False):
        G_prop = p.G
        self._update_edges(G_prop, p.z)
        # draw(G_prop, p.z, "after ipdate")
        self._sample_edges(p)
        p.observation_update()
        if after_interaction:
            interaction_update(p.G, p.z_solution, phase_number=self.phase_number)
        return p.G