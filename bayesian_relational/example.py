from draw.draw import *
from data.initial_data import solution
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LAST_PHASE = 7


def _get_unknown_entries(G):
    n = G.shape[0]
    entries = []
    for i in range(n-3, n):
        for j in range(n):
            entries.append(G[i,j])

    for j in range(n-3, n):
        for i in range(n-3):
            entries.append(G[i,j])
    return entries

#TODO tak naprawde to powinno sie liczyc tylko dla tych co nie znane
def measure(G, G_correct, phase):
    # g = G.flatten()
    # g_correct = G_correct.flatten()
    g = _get_unknown_entries(G)
    g_correct = _get_unknown_entries(G_correct)
    print("G correct")
    print(g_correct)

    print("G")
    print(g)

    print("Precision score: %s" % phase)
    print(precision_score(g_correct, g))

    print("Recall score: %s" % phase)
    print(recall_score(g_correct, g))

    print("F1 score %s" % phase)
    print(f1_score(g_correct, g))



for phase in range(7, LAST_PHASE+1):
    z_init = initial_classes(phase)
    z_correct = correct_classes(phase)
    G_init = initial_graph(phase)
    G, z, scores = run(G_init, z_init)
    interaction_update(G, z_init, phase_number=phase)
    G, z, scores = run(G, z)
    G_correct, z_correct = solution(phase)
    measure(G, G_correct,phase)

    # draw(G, z)

