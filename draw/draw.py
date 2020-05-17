import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm

from bayesian_relational.crp import *
from data.initial_data import *
# ystapieniue do zusuu a zasilek rehabilitacyjyw
def draw(G: np.ndarray, z: np.ndarray, title: str='', save: str=''):
    """
    Draw graph and color its nodes according to z-categorization
    """
    g = nx.from_numpy_array(G, create_using=nx.DiGraph())
    pos = nx.circular_layout(g)
    z_bincount = np.bincount(z)
    K = len(z_bincount)
    viridis = cm.get_cmap('viridis')

    nx.draw_networkx_nodes(g, pos, nodelist=range(len(z)), node_color=z, cmap=viridis)
    nx.draw_networkx_edges(g, pos=pos)
    nx.draw_networkx_labels(g, pos=pos, labels = {i:f'Z{i}' for i in range(len(z))} )
    plt.box(False)
    if title:
        plt.title(title)
    if save:
        plt.savefig(save)
    else:
        plt.show()


if __name__ == '__main__':
    G = solution(1)
    z = correct_classes(1)
    draw(G, z)
