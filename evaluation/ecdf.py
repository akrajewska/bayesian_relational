from multiprocessing import Process, Manager

import numpy as np
from scipy.spatial import distance
from statsmodels.distributions.empirical_distribution import ECDF
from non_reversible_mcmc.mcmc_search_and_score import SearchScore, ManyEdgesStep
from data.initial_data import initial_point
from matplotlib import pyplot
import multiprocessing as mp
import ray

ray.init()
SAMPLE_SIZE = 1000

@ray.remote
def measure(error):
    p_init = initial_point(7)
    ss = SearchScore(alpha=1, beta=(1, 1), step=ManyEdgesStep, max_epochs=10000, p_init=p_init)
    return ss.run(stop_distance=error)[1]

if __name__ == '__main__':
    # pool = mp.Pool(mp.cpu_count())
    # results = [pool.apply(measure, args=(5,)) for i in range(1000)]
    # print(results)
    sample = [measure.remote(2) for i in range(100)]
    ecdf = ECDF(ray.get(sample))
    pyplot.plot(ecdf.x, ecdf.y)
    pyplot.show()