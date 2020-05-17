from non_reversible_mcmc.mcmc_search_and_score import SearchScore, ManyEdgesStep, OneEdgeStep, Point
from data.initial_data import initial_point
from draw.draw import draw
import numpy as np
import ray
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv
ray.init()
SAMPLE_SIZE = 100

@ray.remote
def run_phase(phase_number:int, initial_type:str):
    p_init = initial_point(phase_number, initial_type)
    # draw(p_init.G, p_init.z, 'Init')
    # draw(p_init.observations, p_init.z_solution, 'solution')
    # ss = SearchScore(alpha=1, beta=(1,1), step=OneEdgeStep, max_epochs=1000, p_init=p_init, phase_number=phase_number, after_interaction=False)
    # p_out, n_epocs = ss.run()
    # draw(p_out.G, p_out.z, 'Output before interaction')
    # p_init.G = p_out.G
    # p_init.z = p_out.z
    ss = SearchScore(alpha=1, beta=(1, 1), step=ManyEdgesStep, max_epochs=1000, p_init=p_init, phase_number=phase_number,
                     after_interaction=True)
    p_out, n_epocs = ss.run()
    # print(p_out.z)
    f1, recall, precision = p_out.all_measures()
    # draw(p_out.G, p_out.z, 'Output after interaction')
    return f1, recall, precision



if __name__ == '__main__':
    # run_phase(1, "C")
    #
    #

    f1_array = np.array([])
    output = {}
    field_names = ['input_type', 'phase', 'f1']
    with open('f1_multi.csv', mode='w') as f1_csv:
        with open('recall_multi.csv', mode='w') as recall_csv:
            with open('precision_multi.csv', mode='w') as precision_csv:
                f1_wr = csv.writer(f1_csv, quoting=csv.QUOTE_ALL)
                recall_wr = csv.writer(recall_csv, quoting=csv.QUOTE_ALL)
                precision_wr = csv.writer(precision_csv, quoting=csv.QUOTE_ALL)

                f1_wr.writerow(['initial data', 'nodes number', 'f1'])
                recall_wr.writerow(['initial data', 'nodes number', 'recall'])
                precision_wr.writerow(['initial data', 'nodes number', 'precision'])

                for initial_type in ["A", "B", "C"]:
                    print(initial_type)
                    results = {}
                    for i in range(1,8):
                        sample = [run_phase.remote(i, initial_type) for j in range(500)]
                        results = ray.get(sample)
                        for f1, recall, precision in results:
                            f1_wr.writerow([initial_type, 3*(i+1), f1])
                            recall_wr.writerow([initial_type, 3*(i+1), recall])
                            precision_wr.writerow([initial_type, 3*(i+1), precision])
                        print(f'phase {i}')
    sns.set(style="whitegrid")

    f1 = pd.read_csv('f1_multi.csv')
    recall = pd.read_csv('recall_multi.csv')
    precision = pd.read_csv('precision_multi.csv')

    ax = sns.pointplot(x="nodes number", y="f1", hue="initial data", data=f1, palette="YlGnBu_d")
    ax.set_title('f1')
    plt.show()


    # ax = sns.pointplot(x="nodes number", y="recall", hue="initial data", data=recall)
    # ax.set_title('recall')
    # plt.show()
    #
    #
    # ax = sns.pointplot(x="nodes number", y="precision", hue="initial data", data=precision)
    # ax.set_title('precision')
    # plt.show()

        # ax = sns.pointplot(x="variable", y="value", hue="initial_type",data=df)

    # result = run_phase(6)
    # print(result)

    # sample = [run_phase.remote(3) for i in range(100)]
    # print(ray.get(sample))