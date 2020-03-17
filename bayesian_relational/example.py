from draw.draw import *

LAST_PHASE = 7

solution = solution(7)

for phase in range(1, LAST_PHASE + 1):
    z_init = correct_classes(phase)
    G_init = generate_initial_graph(phase)
    G, z, scores = run(G_init, z_init)
    interaction_update(G, z_init, phase_number=phase)
    G, z, scores = run(G, z)
    draw(G, z)
