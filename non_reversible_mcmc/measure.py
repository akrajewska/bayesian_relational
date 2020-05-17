import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def _get_unknown_entries(G: np.ndarray):
    g = nx.from_numpy_array(G, create_using=nx.DiGraph())
    unknown_nodes = list(g.nodes)[-3:]
    unknown_edges = {}
    no_nodes = G.shape[0]
    for node in unknown_nodes:
        for i in range(no_nodes):
            unknown_edges[(node, i)] = int((node, i) in g.edges)
            unknown_edges[(i, node)] = int((i, node) in g.edges)

    return unknown_edges

def _get_probe_edges(G: np.ndarray, phase_number: int):
    probe_index = 3 * phase_number
    unknown_edges = {}
    no_nodes = G.shape[0]
    g = nx.from_numpy_array(G, create_using=nx.DiGraph())
    for i in range(no_nodes):
        unknown_edges[(probe_index, i)] = int((probe_index, i) in g.edges)
        unknown_edges[(i, probe_index)] = int((i, probe_index) in g.edges)

    return unknown_edges

def measure_run(G: np.ndarray, G_solution: np.ndarray, phase_number: int):
    predicted_edges = _get_unknown_entries(G)

    correct_edges = _get_unknown_entries(G_solution)
    pred = []
    corr = []
    for key in predicted_edges:
        pred.append(predicted_edges[key])
        corr.append(correct_edges[key])
    print(pred)
    print(corr)
    print(f"Precision score phase {phase_number}: {precision_score(corr, pred)}")
    print(f"Recall score phase {phase_number}: {recall_score(corr, pred)}")
    print(f"F1 score phase {phase_number}: {f1_score(corr, pred)}")