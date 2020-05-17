import unittest
from numpy.testing import assert_array_equal
# from bayesian_relational.crp import remove_node, count_edges, reassign_nodes, contradict
import numpy as np
from draw.draw import draw
from data.initial_data import *
from non_reversible_mcmc.point import Point
from non_reversible_mcmc.step_methods import Step
from non_reversible_mcmc.mcmc_search_and_score import SearchScore


class CRPTestCase(unittest.TestCase):

    def test_remove_node(self):
        z = np.array([0, 1, 1, 0, 1])
        step = Step(alpha=1, beta=(1, 1))
        step.reassigned_idx = 1
        z = step._remove_node(z)
        assert_array_equal(z, np.array([0, 1, 0, 1]))

    def test_remove_singleton(self):
        z = np.array([0, 2, 2, 1, 2])
        step = Step(alpha=1, beta=(1, 1))
        step.reassigned_idx = 3
        z = step._remove_node(z)
        assert_array_equal(z, np.array([0, 1, 1, 1]))

    def test_reassign_nodes(self):
        z = np.array([0, 0, 1, 1, 1, 1])
        step = Step(alpha=1, beta=(1, 1))
        step.reassigned_idx = 3
        z = step._categorization_step(z)
        self.assertEqual(len(z), 6)

    def test_count_edges_trivial(self):
        G = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        z = np.array([0, 1, 0])
        ss = SearchScore(alpha=1, beta=(1, 1), step_class=Step, max_epochs=1, p_init=Point(G, z))
        result = ss.count_edges(G, z)
        expected = ([0.], [2.])
        self.assertEqual(expected, result)

    def test_count_edges(self):
        G = np.array([[0., 0., 1., 1., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 1., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 1., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        z = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0])
        expected = ([15.], [3.])
        ss = SearchScore(alpha=1, beta=(1, 1), step_class=Step, max_epochs=1, p_init=Point(G, z))
        result = ss.count_edges(G, z)
        self.assertEqual(expected, result)

    def test_contradicts(self):
        G = np.array(
            [[0., 0., 1., 1., 1., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        )
        z = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0])
        p = Point(G=G, z=z)
        self.assertTrue(Point.contradict(G, z))

    def test_interaction_update(self):
        G = [[0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0.]]

    # def test_hmmm(self):
    #     a = True
    #     G = np.array([[0., 0., 1., 1., 1., 0., 0., 0., 1.,],
    #                  [0., 0., 1., 1., 1., 0., 0., 0., 1.,],
    #                  [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
    #                  [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
    #                  [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
    #                  [0., 0., 1.,1., 1., 0., 0., 0., 0.,],
    #                  [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
    #                  [0., 0., 1., 1., 1., 0., 0., 0., 0.,],
    #                  [0., 0., 0., 0., 0., 0., 0., 0., 0.,]])
    #     SOLUTION, z  = solution(100)
    #     l = G.shape[0]
    #     for i in range(l-3):
    #         for j in range(l-3):
    #             if G[i][j] != SOLUTION[i][j]:
    #                 a = False
    #     return True


if __name__ == '__main__':
    unittest.main()
