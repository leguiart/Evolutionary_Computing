from hklearn_genetic.genetic_algorithm import GeneticAlgorithm
from hklearn_genetic.problem import BinaryRastrigin, BinaryBeale, BinaryHimmelblau, BinaryEggholder, RealRastrigin, RealBeale, RealHimmelblau, RealEggholder
from scipy import signal
from utils import average_list_of_lists, plot_grouped_bar_graph
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import logging.handlers
import os
import math
import itertools as iter

BINARY = False
REAL = True
RUNS = 5

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "simple_ga_tests.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

def plot_fixed_params(fns, params, avg_times, avg_iters, codif):


    #print(matplotlib.get_backend())
    for param in params.keys():
        print(f"Selection method: "f"{param}")
        logging.info(f"Selection method: "f"{param}")
        for p in params[param].keys():
            print(f"Function to optimize: "f"{p}")
            print(f"Parameters: "f"{params[param][p]}")       
            logging.info(f"Function to optimize: "f"{p}")
            logging.info(f"Parameters: "f"{params[param][p]}")
            gens_avg = []
            elapsed_avg = []
            ga_best_li = []
            ga_avg_li = []
            for i in range(RUNS):
                if param[-1] == "E":
                    #elitismo
                    ga = GeneticAlgorithm(pc = params[param][p]["pc"], pm = params[param][p]["pm"], max_iter=params[param][p]["max_iter"], elitism=params[param][p]["elitism"], selection=params[param][p]["selection"])
                else:
                    #sin elitismo
                    ga = GeneticAlgorithm(pc = params[param][p]["pc"], pm = params[param][p]["pm"], max_iter=params[param][p]["max_iter"], selection=params[param][p]["selection"])

                print(f"Run: "f"{i}")
                logging.info(f"Run: "f"{i}")
                start = timer()
                sol, gens = ga.evolve(fns[p], params[param][p]["n_individuals"])
                end = timer()
                elapsed = end - start # Time in seconds, e.g. 5.38091952400282
                elapsed_avg += [elapsed]
                if gens < params[param][p]["max_iter"]:
                    print(f"Global minimum found: "f"{sol}")
                    print(f"after: "f"{gens}"f" generations.")
                    logging.info(f"Global minimum found: "f"{sol}")
                    logging.info(f"after: "f"{gens}"f" generations.")
                else:
                    print(f"Local minimum found: "f"{sol}")
                    print(f"after maximum number of generations ("f"{gens}"f") elapsed.")
                    logging.info(f"Local minimum found: "f"{sol}")
                    logging.info(f"after maximum number of generations ("f"{gens}"f") elapsed.")
                print(f"Elapsed time: {elapsed} [s]")
                logging.info(f"Elapsed time: {elapsed} [s]")
                gens_avg += [gens]

                ga_best_li += [[fns[p].rank - b for b in ga.best]]
                ga_avg_li += [[fns[p].rank - a for a in ga.averages]]
            g_avg = np.average(gens_avg)
            t_avg = np.average(elapsed_avg)
            print(f"Solutions found after an average of "f"{g_avg} iterations")
            logging.info(f"Solutions found after an average of "f"{g_avg} iterations")
            print(f"In an average time of "f"{t_avg} [s]")
            logging.info(f"In an average time of "f"{t_avg} [s]")
            avg_iters[param][p]=(g_avg, np.std(gens_avg))
            avg_times[param][p]=(t_avg, np.std(elapsed_avg))
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
            fig.suptitle(f'{param}, 'f'5 run average', fontsize=16) 
            axs[1].plot(np.arange(int(g_avg)), ga_avg_avg[0:int(g_avg)], color = 'r', label = 'Average')
            axs[1].set_title(p)
            axs[1].set_xlabel('Generations')
            axs[1].set_ylabel('Average fitness') 
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()   
            plt.legend()
            #plt.show()
            plt.savefig(f"{param}"f"_"f"{p}"f"_"f"avg", bbox_inches='tight')

    groups = list(avg_times.keys())
    labels = list(avg_times[groups[0]].keys())
    data_times = []
    data_gens = []
    for k in avg_times.keys():
        data_times += [[val[0] for val in avg_times[k].values()]]
        data_gens += [[val[0]  for val in avg_iters[k].values()]]
    
    plot_grouped_bar_graph('Average time', 'Average time per selection method and function', labels, groups, data_times, f"Average_times_{codif}", 0.1, 1)
    plot_grouped_bar_graph("'Average generations'", 'Average generations per selection method and function', labels, groups, data_gens, f"Average_generations_{codif}", 0.1, 1)

def get_best_parameters(fns, parameters_set, runs):
    for selection_method in parameters_set.keys():
        for function_to_optimize in fns.keys():
            print(f"Starting parameter sweep for optimizing {function_to_optimize}, with selection method: {selection_method}")
            logging.info(f"Starting parameter sweep for optimizing {function_to_optimize}, with selection method: {selection_method}")
            params = get_best_parameter(fns[function_to_optimize], list(parameters_set[selection_method][function_to_optimize].values()), runs)
            for i, param in enumerate(list(parameters_set[selection_method][function_to_optimize].keys())):
                parameters_set[selection_method][function_to_optimize][param] = params[i]
            print(f"Parameters chosen: {parameters_set[selection_method][function_to_optimize]}")
            logging.info(f"Parameters chosen: {parameters_set[selection_method][function_to_optimize]}")


def get_best_parameter(function, params_to_sweep, runs):
    parameters = list(iter.product(*params_to_sweep))
    params_idxs = []
    for j, param in enumerate(parameters):
        iter_avg = 0
        for i in range(runs):
            ga = GeneticAlgorithm(*param[1 : len(param)])
            _, its = ga.evolve(function, param[0])
            iter_avg += its
        iter_avg = int(iter_avg/runs)
        logging.info(f"The following parameters: {param}\nHad a performace of: {iter_avg}")

        params_idxs += [(j, iter_avg)]
    params_idxs.sort(key = lambda t : t[1])
    return parameters[params_idxs[0][0]]

        

if BINARY:
    rast = BinaryRastrigin(n_dim = 2, n_prec=8)
    beale = BinaryBeale(n_prec=8)
    himme = BinaryHimmelblau(n_prec=8)
    egg = BinaryEggholder(n_prec=4)
    fns = {"Rastrigin" : rast, "Beale" : beale, "Himmelblau" : himme, "Eggholder" : egg}
    params = {
    "PS_BINARY":
    {
    "Rastrigin":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(rast.gene_length*2),  0.75/(rast.gene_length*2),  1./(rast.gene_length*2)], "max_iter":[1000],"selection":["proportional"]}, 
    "Beale":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(beale.gene_length*2),  0.75/(beale.gene_length*2),  1./(beale.gene_length*2)], "max_iter":[1000],"selection":["proportional"]},
    "Himmelblau":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(himme.gene_length*2),  0.75/(himme.gene_length*2),  1./(himme.gene_length*2)], "max_iter":[1000],"selection":["proportional"]}, 
    "Eggholder":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(egg.gene_length*2),  0.75/(egg.gene_length*2),  1./(egg.gene_length*2)], "max_iter":[1000],"selection":["proportional"]}
    },
    "PS_E_BINARY":
    {
    "Rastrigin":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(rast.gene_length*2),  0.75/(rast.gene_length*2),  1./(rast.gene_length*2)], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]}, 
    "Beale":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(beale.gene_length*2),  0.75/(beale.gene_length*2),  1./(beale.gene_length*2)], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]},
    "Himmelblau":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(himme.gene_length*2),  0.75/(himme.gene_length*2),  1./(himme.gene_length*2)], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]}, 
    "Eggholder":{"n_individuals" : [500], "pc" :[0.85, 0.9, 0.95], "pm" : [ 0.25/(egg.gene_length*2),  0.75/(egg.gene_length*2),  1./(egg.gene_length*2)], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]}
    },
    "TS_BINARY":
    {
    "Rastrigin":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(rast.gene_length*2),  0.75/(rast.gene_length*2),  1./(rast.gene_length*2)],"max_iter":[1000],"selection":["tournament"]}, 
    "Beale":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(beale.gene_length*2),  0.75/(beale.gene_length*2),  1./(beale.gene_length*2)],"max_iter":[1000],"selection":["tournament"]},
    "Himmelblau":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(himme.gene_length*2),  0.75/(himme.gene_length*2),  1./(himme.gene_length*2)],"max_iter":[1000],"selection":["tournament"]}, 
    "Eggholder":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(egg.gene_length*2),  0.75/(egg.gene_length*2),  1./(egg.gene_length*2)],"max_iter":[1000],"selection":["tournament"]}
    },
    "TS_E_BINARY":
    {
    "Rastrigin":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(rast.gene_length*2),  0.75/(rast.gene_length*2),  1./(rast.gene_length*2)],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]}, 
    "Beale":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(beale.gene_length*2),  0.75/(beale.gene_length*2),  1./(beale.gene_length*2)],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]},
    "Himmelblau":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(himme.gene_length*2),  0.75/(himme.gene_length*2),  1./(himme.gene_length*2)],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]}, 
    "Eggholder":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(egg.gene_length*2),  0.75/(egg.gene_length*2),  1./(egg.gene_length*2)],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]}
    },
    "SUS_BINARY":
    {
    "Rastrigin":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" :[ 0.25/(rast.gene_length*2),  0.75/(rast.gene_length*2),  1./(rast.gene_length*2)],"max_iter":[1000],"selection":["sus"]}, 
    "Beale":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(beale.gene_length*2),  0.75/(beale.gene_length*2),  1./(beale.gene_length*2)],"max_iter":[1000],"selection":["sus"]},
    "Himmelblau":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(himme.gene_length*2),  0.75/(himme.gene_length*2),  1./(himme.gene_length*2)],"max_iter":[1000],"selection":["sus"]}, 
    "Eggholder":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(egg.gene_length*2),  0.75/(egg.gene_length*2),  1./(egg.gene_length*2)],"max_iter":[1000],"selection":["sus"]}
    },
    "SUS_E_BINARY":
    {
    "Rastrigin":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(rast.gene_length*2),  0.75/(rast.gene_length*2),  1./(rast.gene_length*2)],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]}, 
    "Beale":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(beale.gene_length*2),  0.75/(beale.gene_length*2),  1./(beale.gene_length*2)],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]},
    "Himmelblau":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(himme.gene_length*2),  0.75/(himme.gene_length*2),  1./(himme.gene_length*2)],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]}, 
    "Eggholder":{"n_individuals" : [500],"pc" :[0.85, 0.9, 0.95],"pm" : [ 0.25/(egg.gene_length*2),  0.75/(egg.gene_length*2),  1./(egg.gene_length*2)],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]}
    }
    }
    avg_times = {
        "PS_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "PS_E_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_E_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_E_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None}
    }
    avg_iters = {
        "PS_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "PS_E_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_E_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_E_BINARY":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None}
    }
    runs = 3
    print(f"Starting parameter sweep for Binary codification over the following:\n {params}")
    logging.info(f"Starting parameter sweep for Binary codification:\n {params}")
    print(f"Parameter sweep will run each parameter combination for {runs} runs")
    logging.info(f"Parameter sweep will run each parameter combination for {runs} runs")
    get_best_parameters(fns, params, runs)
    print(f"Best parameters for Binary codification after parameter sweep are:\n {params}")
    logging.info(f"Best parameters for Binary codification after parameter sweep are:\n {params}")
    plot_fixed_params(fns, params, avg_times, avg_iters, "BINARY")

if REAL:
    rast = RealRastrigin()
    beale = RealBeale()
    himme = RealHimmelblau()
    egg = RealEggholder()
    fns = {"Rastrigin" : rast, "Beale" : beale, "Himmelblau" : himme, "Eggholder" : egg}
    params = {
        "PS_REAL":
        {
            "Rastrigin":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"]}, 
            "Beale":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"]},
            "Himmelblau":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"]}, 
            "Eggholder":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"]}
        },
        "PS_E_REAL":
        {
            "Rastrigin":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]}, 
            "Beale":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]},
            "Himmelblau":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]}, 
            "Eggholder":{"n_individuals":[500], "pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5], "max_iter":[1000],"selection":["proportional"], "elitism":[0.1, 0.2, 0.3]}
        },
        "TS_REAL":
        {
            "Rastrigin":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"]}, 
            "Beale":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95], "pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"]},
            "Himmelblau":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" :[0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"]}, 
            "Eggholder":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"]}
        },
        "TS_E_REAL":
        {
            "Rastrigin":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]}, 
            "Beale":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]},
            "Himmelblau":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]}, 
            "Eggholder":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["tournament"],"elitism":[0.1, 0.2, 0.3]}
        },
        "SUS_REAL":
        {
            "Rastrigin":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" :[0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"]}, 
            "Beale":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"]},
            "Himmelblau":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"]}, 
            "Eggholder":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"]}
        },
        "SUS_E_REAL":
        {
            "Rastrigin":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]}, 
            "Beale":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]},
            "Himmelblau":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]}, 
            "Eggholder":{"n_individuals":[500],"pc" :[0.85, 0.9, 0.95],"pm" : [0.2, 0.25, 0.5],"max_iter":[1000],"selection":["sus"],"elitism":[0.1, 0.2, 0.3]}
        }
    }
    avg_times = {
        "PS_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "PS_E_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_E_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_E_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None}
    }
    avg_iters = {
        "PS_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "PS_E_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "TS_E_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None},
        "SUS_E_REAL":{"Rastrigin": None, "Beale" : None, "Himmelblau" : None, "Eggholder" : None}
    }
    # avg_times = {
    #     "PS_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "PS_E_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "TS_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "TS_E_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "SUS_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "SUS_E_REAL":{"Himmelblau" : None, "Eggholder" : None}
    # }
    # avg_iters = {
    #     "PS_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "PS_E_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "TS_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "TS_E_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "SUS_REAL":{"Himmelblau" : None, "Eggholder" : None},
    #     "SUS_E_REAL":{"Himmelblau" : None, "Eggholder" : None}
    # }
    runs = 3
    print(f"Starting parameter sweep for Real codification over the following:\n {params}")
    logging.info(f"Starting parameter sweep for Real codification:\n {params}")
    print(f"Parameter sweep will run each parameter combination for {runs} runs")
    logging.info(f"Parameter sweep will run each parameter combination for {runs} runs")
    get_best_parameters(fns, params, runs)
    print(f"Best parameters for Real codification after parameter sweep are:\n {params}")
    logging.info(f"Best parameters for Real codification after parameter sweep are:\n {params}")
    plot_fixed_params(fns, params, avg_times, avg_iters, "REAL")


