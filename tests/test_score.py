from data.initial_data import solution
from draw.draw import draw
from non_reversible_mcmc.mcmc_search_and_score import SearchScore
from non_reversible_mcmc.point import Point
from non_reversible_mcmc.step_methods import Step

if __name__ == '__main__':
    G, z =solution(phase=3)
    point = Point(G,z)
    draw(G, z)
    ss = SearchScore(alpha=1, beta=(1, 1), step_class=Step, max_epochs=1, p_init=Point(G, z))
    ss.score(point)