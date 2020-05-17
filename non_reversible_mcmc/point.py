import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from non_reversible_mcmc import logging

class Point:

    def __init__(self, G, z, observation=None, z_solution=None):
        self.G = G
        self.z = z
        self.dim = len(z)
        if observation is not None:
            self.observations = observation
        if z_solution is not None:
            self.z_solution = z_solution

    @classmethod
    def contradict(cls, G: np.ndarray, z: np.ndarray):
        for i in range(len(z)):
            for j in range(len(z)):
                if z[i] <= z[j]:
                    # contradicts
                    if G[j, i] == 1:
                        return True
        return False

    def z_admissible(self, z: np.ndarray):
        if self.contradict(self.observations, z[:self.dim-3]):
            logging.info('Categorization not admissible')
            return False

        return True

    def observation_update(self):
        for j in range(self.dim - 3):
            for l in range(self.dim - 3):
                self.G[j, l] = self.observations[j, l]

    def _get_unknown_entries(self, G: np.ndarray):
        g = nx.from_numpy_array(G, create_using=nx.DiGraph())
        unknown_nodes = list(g.nodes)[-1:]
        unknown_edges = {}
        no_nodes = G.shape[0]
        for node in unknown_nodes:
            for i in range(no_nodes):
                unknown_edges[(node, i)] = int((node, i) in g.edges)
                unknown_edges[(i, node)] = int((i, node) in g.edges)

        return unknown_edges

    def f1_measure(self):
        predicted_edges = self._get_unknown_entries(self.G)
        correct_edges = self._get_unknown_entries(self.observations)
        pred = [predicted_edges[key] for key in predicted_edges]
        corr = [correct_edges[key] for key in correct_edges]
        return f1_score(corr, pred)

    def recall(self):
        predicted_edges = self._get_unknown_entries(self.G)
        correct_edges = self._get_unknown_entries(self.observations)
        pred = [predicted_edges[key] for key in predicted_edges]
        corr = [correct_edges[key] for key in correct_edges]
        return recall_score(corr, pred)

    def precission(self):
        predicted_edges = self._get_unknown_entries(self.G)
        correct_edges = self._get_unknown_entries(self.observations)
        pred = [predicted_edges[key] for key in predicted_edges]
        corr = [correct_edges[key] for key in correct_edges]
        return precision_score(corr, pred)

    def all_measures(self):
        predicted_edges = self._get_unknown_entries(self.G)
        correct_edges = self._get_unknown_entries(self.observations)
        pred = [predicted_edges[key] for key in predicted_edges]
        corr = [correct_edges[key] for key in correct_edges]
        return f1_score(corr, pred), recall_score(corr, pred), precision_score(corr, pred)