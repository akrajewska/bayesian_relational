import numpy as np
from non_reversible_mcmc.point import Point
from non_reversible_mcmc.step_methods import Step, ManyEdgesStep, OneEdgeStep
from scipy.special import gamma, factorial, beta
from itertools import combinations
from typing import Type
from numpy import random
from data.initial_data import initial_point, solution
from draw.draw import draw
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from copy import deepcopy
from non_reversible_mcmc import logging
from scipy.spatial import distance
from scipy.special import comb
from math import log, log10

class SearchScore:

    max_score = 0
    max_output = None
    scores = None
    p_init = None

    def __init__(self, alpha: float, beta: tuple, step: Type[Step], max_epochs: int, p_init: Point, phase_number: int, after_interaction: bool):
        self.alpha = alpha
        self.beta = beta
        self.step = step(alpha=self.alpha, beta=self.beta, phase_number=phase_number)
        #TODO to chyba niepotrzebne
        if not self.scores:
            self.scores = []
        self.max_epochs = max_epochs
        self.p_init = p_init
        self._set_solution_score()
        self.measures = []
        self.after_interaction = after_interaction

    def _set_solution_score(self):
        if self.p_init.z_solution is not None and self.p_init.observations is not None:
            solution_point = Point(G=self.p_init.observations, z=self.p_init.z_solution)
            logging.debug('Scoring solution point')
            self.solution_score = self.score(solution_point)


    def z_nodes_permutions(self, z_bincount: np.ndarray):
        """
        :param z_bincount: vector counting occurrences of each class
        :return: number of nodes permutations underwhich z can be obtained
        """
        return np.prod(factorial(z_bincount))

    def count_edges(self,G: np.ndarray, z: np.ndarray, z_bincount: np.ndarray = None, K: int = None):
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

    def conditional_probability(self, G: np.ndarray, z: np.ndarray, _beta: tuple, z_bincount: np.ndarray = None,
                                K: int = None):
        """
        Probability of graph G given z classification
        """
        if z_bincount is None:
            z_bincount = np.bincount(z)
        if K is None:
            K = len(z_bincount)
        present_edges, absent_edges = self.count_edges(G, z, z_bincount, K)
        beta1, beta2 = _beta
        p_G_given_z = np.prod([beta(beta1 + present_edges[i], beta2 + absent_edges[i])/beta(beta1, beta2) for i in range(len(present_edges))])
        return p_G_given_z

    def score(self, p: Point):
        """
        Score function is given by a joint probability of pair (G, z)
        :return: score value: float
        """
        G = p.G
        z = p.z
        alpha = self.alpha
        beta = self.beta

        if Point.contradict(G, z):
            return 0

        logging.debug(f'Scoring categorization {z}')
        z_bincount = np.bincount(z)
        # number of classes
        K = len(z_bincount)
        d = G.shape[1]
        scalar = 1 / factorial(d) * gamma(alpha) / gamma(alpha + d) * alpha ** K
        A = self.z_nodes_permutions(z_bincount)
        logging.debug(f'Number of nodes permutations given z {A}')
        B = self.conditional_probability(G, z, beta, z_bincount, K)
        logging.debug(f'Conditional probability for G given z {B}')
        _score = scalar * A * A * B
        logging.debug(f'Score value {_score}')
        # draw(G, z, f'score {_score}')
        return _score

    def accept(self, current_score: float, last_score: float):
        if last_score == 0:
            return 1
        return min(1, current_score / last_score)

    def _step(self, p: Point):
        p_prop = self.step.step(p, after_interaction=self.after_interaction)
        current_score = self.score(p_prop)
        if current_score == 0:
            return p
        r = random.uniform()
        last_score = self.scores[-1]
        _alpha = self.accept(current_score, last_score)
        if r < _alpha:
            # draw(p_prop.G, p_prop.z)
            if current_score > self.max_score:
                self.max_score = current_score
                self.p_max = deepcopy(p_prop)
                self.scores.append(current_score)
            return p_prop
        return p

    def _check_stop(self, stop_distance):
        last_score = self.scores[-1]
        solution_score = self.solution_score
        logging.debug(f'Checking distance for {last_score} and {solution_score}')
        logging.debug(f'Distance from solution {abs(log10(last_score)-log10(solution_score))}')
        if distance.euclidean(log10(last_score), log10(solution_score)) <= stop_distance:
            return True
        return False

    def run(self, stop_distance: float=0):
        #TODO wyniesc do init
        score_init = self.score(self.p_init)
        self.scores = [score_init]
        self.max_score = score_init
        p = self.p_init
        self.p_max = self.p_init
        for epoch in range(self.max_epochs):
            p = self._step(p)
            # self.measures.append(p.f1_measure())
            if stop_distance:
                if self._check_stop(stop_distance):
                    logging.debug(f'Output score {self.score(self.p_max)}')
                    return self.p_max, epoch+1
        if stop_distance:
            logging.debug(f'Output score {self.score(self.p_max)}')
            return self.p_max,np.Inf
        # logging.debug(f'F1 score of solution {self.p_max.f1_measure()}')
        # print(self.scores)
        logging.debug(f'Output score {self.score(self.p_max)}')
        # p_solution = Point(G_s, z_s)
        return self.p_max, epoch+1



def dag_number(n):
    dag_numbers = np.zeros(n)
    if n == 1:
        return 1
    else:
        counter = 0
        for i in range(1, n+1):
            j = n-i
            if dag_numbers[j] > 0:
                _dag_number = dag_numbers[j]
            else:
                _dag_number = dag_number(j)
                dag_numbers[j] = _dag_number

            counter += (-1)**(i-1) * comb(n, i) * 2**(i*(n-i)) * _dag_number

        return counter



if __name__ == '__main__':
    print(dag_number(10))

    initial_point = initial_point(10)
    ss = SearchScore(alpha=1, beta=(1, 1), step_class=ManyEdgesStep, max_epochs=10000, p_init=initial_point, phase_number=7)
    # p_result = ss.run()
    # draw(p_result.G, p_result.z)