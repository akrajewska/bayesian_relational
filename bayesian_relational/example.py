from bayesian_relational.crp import *
from data.initial_data import *

G_init = generate_initial_graph(1)
z_init = generate_initial_classes(1)

run(G_init, z_init, 1000, 1, 1)
