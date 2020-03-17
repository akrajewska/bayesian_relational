import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm

from bayesian_relational.crp import *
from data.initial_data import *


def draw(G: np.ndarray, z: np.ndarray):
    """
    Draw graph and color its nodes according to z-categorization
    """
    g = nx.from_numpy_array(G, create_using=nx.DiGraph())
    pos = nx.circular_layout(g)
    z_bincount = np.bincount(z)
    K = len(z_bincount)
    viridis = cm.get_cmap('viridis', K)
    for _class in range(K):
        _class_nodes = np.argwhere(z == _class).flatten()
        nx.draw_networkx_nodes(g, pos, nodelist=_class_nodes, node_color=viridis(_class))
    nx.draw_networkx_edges(g, pos=pos)
    plt.show()


if __name__ == '__main__':
    G = solution(7)
    z = correct_classes(7)
    draw(G, z)
    G_init = generate_initial_graph(7)
    G, z, scores = run(G_init, z)
    draw(G, z)
