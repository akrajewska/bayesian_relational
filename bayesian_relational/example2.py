from bayesian_relational.crp import *
from data.initial_data import *
from evaluation.ecdf import *

G = solution(20)
G_init = initial_graph(20)
z = initial_classes(20)

error = 10 ** -8
initial = G_init, z, []
solution = G_init, z

runtimes = meausure(step, initial, score, solution, error, sample_size=SAMPLE_SIZE, max_runtime=MAX_RUNTIME)
print(list(runtimes))
