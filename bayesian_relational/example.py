from bayesian_relational.crp import *
from data.initial_data import *

LAST_PHASE = 7

# G_init = generate_initial_graph(7)
# z_init = generate_solution_classes(7)
#
solution = solution(7)
#
# G, *rest = run(G_init, z_init, 1000, 1, 1)
#
# p = metrics.precision_score(solution.flatten(), G.flatten())
# r = metrics.recall_score(solution.flatten(), G.flatten())
# f = metrics.f1_score(solution.flatten(), G.flatten())


if __name__ == '__main__':
    z_correct = generate_solution_classes(LAST_PHASE)
    for phase_number in range(1, LAST_PHASE + 1):
        G_init = generate_initial_graph(phase_number)
        z_init = generate_solution_classes(phase_number)
        G, z, scores = run(G_init, z_init, 1000, 1, 1)
        interaction_update(G, z_correct, phase_number)

    print(scores)

    print(score(G, z_correct))
    print(score(solution, z_correct))
