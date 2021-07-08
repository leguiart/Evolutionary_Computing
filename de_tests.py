from numpy.lib.function_base import diff
from hklearn_genetic.differential_evolution import differential_evolution
from hklearn_genetic.pso import RealRastriginPSO, RealBealePSO, RealHimmelblauPSO, RealEggholderPSO
from utils import average_list_of_lists, plot_superposed
from timeit import default_timer as timer
import logging
import logging.handlers
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools as iter


handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "de_tests.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

RUNS = 10
DO_PARAM_SWEEP = False

def plot_fixed_params(fns, params, avg_times, avg_iters, pops):

    for pop in pops:
        #print(matplotlib.get_backend())
        for p in params.keys():
            print(f"Function to optimize: "f"{p}")
            print(f"Parameters: "f"{params[p]}")       
            logging.info(f"Function to optimize: "f"{p}")
            logging.info(f"Parameters: "f"{params[p]}")
            gens_avg = []
            elapsed_avg = []
            ga_best_li = []
            ga_avg_li = []
            for i in range(RUNS):


                print(f"Run: "f"{i}")
                logging.info(f"Run: "f"{i}")
                start = timer()
                pars = list(params[p].values())
                ps = [fns[p]] + [pop] + pars[1: len(pars)]
                sol, gens, best, averages = differential_evolution(*ps)
                end = timer()
                elapsed = end - start # Time in seconds, e.g. 5.38091952400282
                elapsed_avg += [elapsed]
                if gens < params[p]["max_iter"]:
                    print(f"Global minimum found: "f"{sol}")
                    print(f"after: "f"{gens}"f" generations.")
                    logging.info(f"Global minimum found: "f"{sol}")
                    logging.info(f"after: "f"{gens}"f" generations.")
                print(f"Elapsed time: {elapsed} [s]")
                logging.info(f"Elapsed time: {elapsed} [s]")
                gens_avg += [gens]

                ga_best_li += [[b[0] for b in best]]
                ga_avg_li += [averages]
            g_avg = np.average(gens_avg)
            t_avg = np.average(elapsed_avg)
            print(f"Solutions found after an average of "f"{g_avg} iterations")
            logging.info(f"Solutions found after an average of "f"{g_avg} iterations")
            print(f"In an average time of "f"{t_avg} [s]")
            logging.info(f"In an average time of "f"{t_avg} [s]")
            # avg_iters[p]=(g_avg, np.std(gens_avg))
            # avg_times[p]=(t_avg, np.std(elapsed_avg))
            avg_iters[p]+=[g_avg]
            avg_times[p]+=[t_avg]
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            for i, data in enumerate(ga_best_li):
                if len(data) <= int(g_avg):
                    #axs[0].plot(np.arange(len(data)), data, color = colors[i] ,label = f'Run: {i}')
                    axs[0].plot(np.arange(len(data)), data, color = '0.8' )
                else:
                    #axs[0].plot(np.arange(int(gens_avg)), data[0 : int(gens_avg)], color = colors[i], label = f'Run: {i}')
                    axs[0].plot(np.arange(int(g_avg)), data[0 : int(g_avg)], color = '0.8')
            for i, data in enumerate(ga_avg_li):
                if len(data) <= int(g_avg):
                    #axs[1].plot(np.arange(len(data)), data, color = colors[i], label = f'Run: {i}')   
                    axs[1].plot(np.arange(len(data)), data, color = '0.8')   
                else:
                    #axs[1].plot(np.arange(int(gens_avg)), data[0 : int(gens_avg)], color = colors[i], label = f'Run: {i}')
                    axs[1].plot(np.arange(int(g_avg)), data[0 : int(g_avg)], color ='0.8')
            ga_best_avg = average_list_of_lists(ga_best_li)
            ga_avg_avg = average_list_of_lists(ga_avg_li)
            axs[0].plot(np.arange(int(g_avg)), ga_best_avg[0:int(g_avg)], color = 'r', label = 'Average')
            axs[0].set_title(p)
            axs[0].set_xlabel('Generations')
            axs[0].set_ylabel('Best fitness')
            fig.suptitle(f'{p}, 'f'5 run average, for n_individuals = {pop}', fontsize=16) 
            axs[1].plot(np.arange(int(g_avg)), ga_avg_avg[0:int(g_avg)], color = 'r', label = 'Average')
            axs[1].set_title(p)
            axs[1].set_xlabel('Generations')
            axs[1].set_ylabel('Average fitness') 
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()   
            plt.legend()
            #plt.show()
            plt.savefig(f"differential_evo_{p}_{pop}", bbox_inches='tight')

    # groups = list(avg_times.keys())
    # labels = list(avg_times[groups[0]].keys())
    # data_times = []
    # data_gens = []
    # for k in avg_times.keys():
    #     data_times += [[val[0] for val in avg_times[k].values()]]
    #     data_gens += [[val[0]  for val in avg_iters[k].values()]]
    
    # plot_grouped_bar_graph('Average time', 'Average time per selection method and function', labels, groups, data_times, f"Average_times_{codif}", 0.1, 1)
    # plot_grouped_bar_graph("'Average generations'", 'Average generations per selection method and function', labels, groups, data_gens, f"Average_generations_{codif}", 0.1, 1)


def get_best_parameters(fns, parameters_set, runs):
    for function_to_optimize in fns.keys():
        print(f"Starting parameter sweep for optimizing {function_to_optimize}, with differential evolution")
        logging.info(f"Starting parameter sweep for optimizing {function_to_optimize}, with differential evolution")
        params = get_best_parameter(fns[function_to_optimize], list(parameters_set[function_to_optimize].values()), runs)
        for i, param in enumerate(list(parameters_set[function_to_optimize].keys())):
            parameters_set[function_to_optimize][param] = params[i]
        print(f"Parameters chosen: {parameters_set[function_to_optimize]}")
        logging.info(f"Parameters chosen: {parameters_set[function_to_optimize]}")


def get_best_parameter(function, params_to_sweep, runs):
    parameters = list(iter.product(*params_to_sweep))
    params_idxs = []
    for j, param in enumerate(parameters):
        iter_avg = 0
        for i in range(runs):
            p = [function] + list(param)
            _, its, _, _ = differential_evolution(*p)
            iter_avg += its
        iter_avg = int(iter_avg/runs)
        logging.info(f"The following parameters: {param}\nHad a performance of: {iter_avg}")

        params_idxs += [(j, iter_avg)]
    params_idxs.sort(key = lambda t : t[1])
    return parameters[params_idxs[0][0]]


rast =  RealRastriginPSO(n_dim=2)
beale = RealBealePSO()
himme = RealHimmelblauPSO()
egg = RealEggholderPSO()
pops = [20, 50, 100, 200, 500]
fns = {"Rastrigin" : rast, "Beale" : beale, "Himmelblau" : himme, "Eggholder" : egg}
avg_times = {"Rastrigin": [], "Beale" : [], "Himmelblau" : [], "Eggholder" : []}
avg_iters = {"Rastrigin": [], "Beale" : [], "Himmelblau" : [], "Eggholder" : []}
if DO_PARAM_SWEEP:

    params = {"Rastrigin":{"n_individuals" : pops, "pc" :[0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "F" : [0.5, 1, 1.5, 2.], "max_iter":[1000]}, 
        "Beale":{"n_individuals" : pops, "pc" :[0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "F" : [0.5, 1, 1.5, 2.], "max_iter":[1000]},
        "Himmelblau":{"n_individuals" : pops, "pc" :[0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "F" : [0.5, 1, 1.5, 2.], "max_iter":[1000]}, 
        "Eggholder":{"n_individuals" : pops, "pc" :[0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "F" : [0.5, 1, 1.5, 2.], "max_iter":[1000]}}
    runs = 5
    print(f"Starting parameter sweep for Differential Evolution over the following:\n {params}")
    logging.info(f"Starting parameter sweep for Differential Evolution over the following:\n {params}")
    print(f"Parameter sweep will run each parameter combination for {runs} runs")
    logging.info(f"Parameter sweep will run each parameter combination for {runs} runs")
    get_best_parameters(fns, params, runs)
    print(f"Best parameters for Differential Evolution after parameter sweep are:\n {params}")
    logging.info(f"Best parameters for Differential Evolution after parameter sweep are:\n {params}")
    plot_fixed_params(fns, params, avg_times, avg_iters, pops)
else:
    params ={
        'Rastrigin': {'n_individuals': 100, 'pc': 0.75, 'F': 0.5, 'max_iter': 1000}, 
        'Beale': {'n_individuals': 200, 'pc': 0.9, 'F': 1, 'max_iter': 1000}, 
        'Himmelblau': {'n_individuals': 500, 'pc': 0.9, 'F': 0.5, 'max_iter': 1000}, 
        'Eggholder': {'n_individuals': 200, 'pc': 0.95, 'F': 1, 'max_iter': 1000}}
    plot_fixed_params(fns, params, avg_times, avg_iters, pops)

avg_iters_values = list(avg_iters.values())
avg_iters_labels = list(avg_iters.keys())
avg_times_values = list(avg_times.values())
avg_times_labels = list(avg_times.keys())
plot_superposed(np.array(pops), avg_iters_values, x_label="Number of individuals", y_label="Number of iterations", sup_title=f"Average number of iterations per number of individuals on {RUNS} runs", plot_labels = avg_iters_labels, store=True, store_path=f"de_iterations_vs_particles", show=False)
plot_superposed(np.array(pops), avg_times_values, x_label="Number of individuals", y_label="Time [s]", sup_title=f"Average time per number of individuals on {RUNS} runs", plot_labels = avg_iters_labels, store=True, store_path=f"de_time_vs_particles", show=False)