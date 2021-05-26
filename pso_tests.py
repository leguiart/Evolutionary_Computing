from hklearn_genetic.genetic_algorithm import GeneticAlgorithm
from hklearn_genetic.problem import BinaryRastrigin, BinaryBeale, BinaryHimmelblau, BinaryEggholder, RealRastrigin, RealBeale, RealHimmelblau, RealEggholder
from hklearn_genetic.pso import pso, RealRastriginPSO, RealBealePSO, RealHimmelblauPSO, RealEggholderPSO
from scipy import signal
from utils import average_list_of_lists, plot_superposed, plot_superposed_multiple_xs
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import logging.handlers
import os
import math
import time

RUNS = 5
# BINARY = False
# REAL = True

# if BINARY:
#     rast = BinaryRastrigin(n_dim = 2, n_prec=8)
#     beale = BinaryBeale(n_prec=8)
#     himme = BinaryHimmelblau(n_prec=8)
#     egg = BinaryEggholder(n_prec=4)

# if REAL:
#     rast = RealRastrigin()
#     beale = RealBeale()
#     himme = RealHimmelblau()
#     egg = RealEggholder()

rast =  RealRastriginPSO(n_dim=2)
beale = RealBealePSO()
himme = RealHimmelblauPSO()
egg = RealEggholderPSO()
params = {"Rastrigin": [rast, 100, 0.25, 0.3, 0.7, 0.5, .1, 0.001], "Beale" : [beale, 100, 0.25, 0.3, 0.7, 0.5, .1, 0.001], "Himmelblau" : [himme, 100, 0.25, 0.3, 0.7, 0.5, .1, 0.001], "Eggholder" : [egg, 100, 0.5, 0.3, 0.7, 0.75, 0.0001, 0.001]}
best_knownss = {"Rastrigin": [], "Beale" : [], "Himmelblau" : [], "Eggholder" : []}
p_best_knownss = {"Rastrigin": [], "Beale" : [], "Himmelblau" : [], "Eggholder" : []}
avg_times = {"Rastrigin": [], "Beale" : [], "Himmelblau" : [], "Eggholder" : []}
avg_iters = {"Rastrigin": [], "Beale" : [], "Himmelblau" : [], "Eggholder" : []}

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "pso_tests.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

pops = [25, 50, 100, 200, 500, 1000]
#pops = [100, 300]
#pops = [100]
# print(pso(100, rast, 1000, 0.5, 0.1, 0.1, 0.8, 1, 0))
# print(pso(100, beale, 1000, 0.1, 0.1, 0.1, 0.5, 1, 0))
# print(pso(100, himme, 1000, 0.1, 0.1, 0.1, 0.5, 1, 0))
# sol, _, pop = pso(100, egg, 5000, 0.1, 0.1, 0.1, 0.5, 0.1)
# print(sol)
# print(pop)
#print(pso(100, egg, 5000, 0.1, 0.5, 0.5, 0.8, 0.1, 0.1))
#print(pso(300, egg, 5000, 0.5, 0.1, 0.1, 0.5, 0.01, 0.1))

for pop in pops:
    for key, param in params.items():
        iter_avg = 0
        elapsed_sum = 0
        best_knowns = []
        p_best_knowns = []
        print(f"Function to optimize: "f"{key}")
        print(f"Parameters: "f"{param}")       
        logging.info(f"Function to optimize: "f"{key}")
        logging.info(f"Parameters: "f"{param}")
        for i in range(RUNS):
            print(f"Run: "f"{i}")
            logging.info(f"Run: "f"{i}")
            start = timer()
            sol, its, _, bks, pbks = pso(pop, param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7]) 
            end = timer()
            elapsed = end - start # Time in seconds, e.g. 5.38091952400282
            elapsed_sum += elapsed
            #best_knowns += [(bks, '0.8')]
            #p_best_knowns += [(pbks, '0.8')]
            best_knowns += [bks]
            p_best_knowns += [pbks]
            iter_avg+=its
            if its < param[1]:
                print(f"Global minimum found: "f"{sol}")
                print(f"after: "f"{its}"f" iterations.")
                logging.info(f"Global minimum found: "f"{sol}")
                logging.info(f"after: "f"{its}"f" iterations.")
            else:
                print(f"Local minimum found: "f"{sol}")
                print(f"after maximum number of iterations ("f"{its}"f") elapsed.")
                logging.info(f"Local minimum found: "f"{sol}")
                logging.info(f"after maximum number of iterations ("f"{its}"f") elapsed.")
            print(f"Elapsed time: {elapsed} [s]")
            logging.info(f"Elapsed time: {elapsed} [s]")
        iter_avg = max(math.ceil(iter_avg/RUNS), 2)
        avg_iters[key]+=[iter_avg]
        avg_times[key]+=[elapsed_sum/RUNS]
        print(f"Solutions found after an average of "f"{iter_avg} iterations")
        logging.info(f"Solutions found after an average of "f"{iter_avg} iterations")
        print(f"In an average time of "f"{elapsed_sum/RUNS} [s]")
        logging.info(f"In an average time of "f"{elapsed_sum/RUNS} [s]")
        avg_best_knowns = (list(average_list_of_lists(best_knowns)[0 : iter_avg]), "r")
        avg_p_best_knowns = (list(average_list_of_lists(p_best_knowns)[0 : iter_avg]), "r")

        for i in range(len(best_knowns)):
            best_knowns[i] = best_knowns[i][0 : iter_avg]
            p_best_knowns[i] = p_best_knowns[i][0 : iter_avg]
            best_knowns[i] = (best_knowns[i], "0.8")
            p_best_knowns[i] = (p_best_knowns[i], "0.8")

        best_knowns += [avg_best_knowns]
        p_best_knowns += [avg_p_best_knowns]
        best_knownss[key]+=[best_knowns]
        p_best_knownss[key]+=[avg_p_best_knowns]

for k, v in best_knownss.items():
    for i in range(len(v)):
        x_s = [np.arange(len(v[i][j][0])) for j in range(len(v[i]))]
        plot_superposed_multiple_xs(x_s, v[i], x_label="Iterations", y_label="Function value", sup_title= f"Best per iteration for {k} with {pops[i]} particles", store=True, store_path=f"pso_swarm_best_{k}_{pops[i]}")
        plot_superposed_multiple_xs(x_s, v[i], x_label="Iterations", y_label="Function value", sup_title= f"Average per iteration for {k} with {pops[i]} particles", store=True, store_path=f"pso_particle_best_{k}_{pops[i]}")

avg_iters_values = list(avg_iters.values())
avg_iters_labels = list(avg_iters.keys())
avg_times_values = list(avg_times.values())
avg_times_labels = list(avg_times.keys())
plot_superposed(np.array(pops), avg_iters_values, x_label="Number of particles", y_label="Number of iterations", sup_title=f"Average number of iterations per number of particles on {RUNS} runs", plot_labels = avg_iters_labels, store=True, store_path=f"pso_iterations_vs_particles")
plot_superposed(np.array(pops), avg_times_values, x_label="Number of particles", y_label="Time [s]", sup_title=f"Average time per number of particles on {RUNS} runs", plot_labels = avg_iters_labels, store=True, store_path=f"pso_time_vs_particles")