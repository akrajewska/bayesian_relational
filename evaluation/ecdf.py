from multiprocessing import Process, Manager

import numpy as np
from scipy.spatial import distance
from statsmodels.distributions.empirical_distribution import ECDF

SAMPLE_SIZE = 200
MAX_RUNTIME = 1000


def stop_condition(output_score: float, correct_score: float, error: float, epoch: int, max_iterations: int = 100):
    if distance.euclidean(output_score, correct_score) <= error:
        return True
    if epoch >= max_iterations:
        return True
    return False


def measure_single_run(step, initial, run_times, score_function, solution_score, error, max_runtime):
    run = True
    while run:
        epoch = 1
        result = step(*initial)
        # TODO uniezaleznienie
        G, z, scores = result
        result_score = score_function(G, z)
        if stop_condition(result_score, solution_score, error, epoch):
            run = False
            measure = epoch
            if epoch > max_runtime:
                measure = np.Inf
    run_times.append(measure)


def meausure(step, initial, score_function, solution, error, sample_size=SAMPLE_SIZE, max_runtime=MAX_RUNTIME):
    solution_score = score_function(*solution)
    manager = Manager()
    run_times = manager.list()
    for i in range(sample_size):
        processes = []
        for j in range(5):
            p = Process(target=measure_single_run,
                        args=(step, initial, run_times, score_function, solution_score, error, max_runtime))
            p.start()
            processes.append(p)
        for l in processes:
            l.join()
    return ECDF(np.array(run_times))


def ecdf(runtimes: list):
    pass
