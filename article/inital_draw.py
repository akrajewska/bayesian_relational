from data.initial_data import initial_point
from draw.draw import draw
from non_reversible_mcmc.mcmc_search_and_score import SearchScore, Point, Step
import os

ARTICLE_DIR = '/home/antonina/learning_casual/IEEEtran/graph'


p_init = initial_point(3)
ss = SearchScore(alpha=1, beta=(1, 1), step=Step, max_epochs=1, p_init=p_init, phase_number=3, after_interaction=False)
_score = ss.score(p_init)
draw(p_init.G, p_init.z, title='', save=os.path.join(ARTICLE_DIR, 'initial_all.png'))