from bayesian_relational.crp import *
from data.initial_data import *
from evaluation.ecdf import *

G = solution(5)
G_init = generate_initial_graph(5)
z = correct_classes(5)

error = 10 ** -1
initial = G_init, z, []
solution = G_init, z

runtimes = meausure(step, initial, score, solution, error, sample_size=SAMPLE_SIZE, max_runtime=MAX_RUNTIME)
print(runtimes)
