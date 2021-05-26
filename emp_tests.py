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

BINARY = True
REAL = True
RUNS = 5

def start(fns, params, avg_times, avg_iters, codif):
    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "simple_ga_tests.log"))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root.addHandler(handler)

    print(matplotlib.get_backend())
    colors = ['b', 'g', 'y', 'c', 'k']
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
                # print(f"Solution(s) found: "f"{sol}")
                # logging.info(f"Solution(s) found: "f"{sol}")
                # fig, axs = plt.subplots(2, 1, constrained_layout=True)
                # axs[0].plot(np.arange(gens), ga.best)
                # axs[0].set_title(f'{p}, Run: 'f'{i}')
                # axs[0].set_xlabel('Generations')
                # axs[0].set_ylabel('Best fitness')
                # fig.suptitle(param, fontsize=16) 
                # axs[1].plot(np.arange(gens), ga.averages)
                # axs[1].set_title(f'{p}, Run: 'f'{i}')
                # axs[1].set_xlabel('Generations')
                # axs[1].set_ylabel('Average fitness') 
                # manager = plt.get_current_fig_manager()
                # manager.window.showMaximized()   
                #plt.savefig(f"{param}"f"_"f"{p}"f"_"f"{i}", bbox_inches='tight')
                #plt.show()
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


if BINARY:
    rast = BinaryRastrigin(n_dim = 2, n_prec=8)
    beale = BinaryBeale(n_prec=8)
    himme = BinaryHimmelblau(n_prec=8)
    egg = BinaryEggholder(n_prec=4)
    fns = {"Rastrigin" : rast, "Beale" : beale, "Himmelblau" : himme, "Eggholder" : egg}
    params = {
    "PS_BINARY":
    {
        "Rastrigin":{"n_individuals" : 500, "pc" : 0.95, "pm" : 0.5/(rast.gene_length*2), "max_iter":1000,"selection":"proportional"}, 
        "Beale":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(beale.gene_length*2), "max_iter":1000,"selection":"proportional"},
        "Himmelblau":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(himme.gene_length*2), "max_iter":1000,"selection":"proportional"}, 
        "Eggholder":{"n_individuals" : 500, "pc" : 0.95, "pm" : 0.25/(egg.gene_length*2), "max_iter":1000,"selection":"proportional"}
    },
    "PS_E_BINARY":
    {
        "Rastrigin":{"n_individuals" : 500, "pc" : 0.9, "pm" : .5/(rast.gene_length*2), "max_iter":1000, "elitism":0.1,"selection":"proportional"}, 
        "Beale":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(beale.gene_length*2), "max_iter":1000, "elitism":0.1,"selection":"proportional"},
        "Himmelblau":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(himme.gene_length*2), "max_iter":1000, "elitism":0.1,"selection":"proportional"}, 
        "Eggholder":{"n_individuals" : 500, "pc" : 0.95, "pm" : 1./(egg.gene_length*2), "max_iter":1000, "elitism":0.2,"selection":"proportional"}
    },
    "TS_BINARY":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : .5/(rast.gene_length*2),"max_iter":1000,"selection":"tournament"}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : .25/(beale.gene_length*2),"max_iter":1000,"selection":"tournament"},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : 0.25/(himme.gene_length*2),"max_iter":1000,"selection":"tournament"}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : .25/(egg.gene_length*2),"max_iter":1000,"selection":"tournament"}
    },
    "TS_E_BINARY":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : .5/(rast.gene_length*2),"max_iter":1000,"selection":"tournament","elitism":0.1}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : 25./(beale.gene_length*2),"max_iter":1000,"selection":"tournament","elitism":0.1},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : .25/(himme.gene_length*2),"max_iter":1000,"selection":"tournament","elitism":0.1}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":1000,"selection":"tournament","elitism":0.2}
    },
    "SUS_BINARY":
    {
        "Rastrigin":{"n_individuals" : 500,"pc" : 0.9,"pm" :.5/(rast.gene_length*2),"max_iter":1000,"selection":"sus"}, 
        "Beale":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(beale.gene_length*2),"max_iter":1000,"selection":"sus"},
        "Himmelblau":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":1000,"selection":"sus"}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":1000,"selection":"sus"}
    },
    "SUS_E_BINARY":
    {
        "Rastrigin":{"n_individuals" : 500,"pc" : 0.9,"pm" : .5/(rast.gene_length*2),"max_iter":1000,"selection":"sus","elitism":0.1}, 
        "Beale":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(beale.gene_length*2),"max_iter":1000,"selection":"sus","elitism":0.1},
        "Himmelblau":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":1000,"selection":"sus","elitism":0.1}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":1000,"selection":"sus","elitism":0.2}
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
    start(fns, params, avg_times, avg_iters, "BINARY")
if REAL:
    rast = RealRastrigin()
    beale = RealBeale()
    himme = RealHimmelblau()
    egg = RealEggholder()
    fns = {"Rastrigin" : rast, "Beale" : beale, "Himmelblau" : himme, "Eggholder" : egg}
    params = {
        "PS_REAL":
        {
            "Rastrigin":{"n_individuals" : 500, "pc" : 0.9, "pm" : 1./(2), "max_iter":1000,"selection":"proportional"}, 
            "Beale":{"n_individuals" : 500, "pc" : 0.9, "pm" : 1./(2), "max_iter":1000,"selection":"proportional"},
            "Himmelblau":{"n_individuals" : 500, "pc" : 0.9, "pm" : 1./(2), "max_iter":1000,"selection":"proportional"}, 
            "Eggholder":{"n_individuals" : 500, "pc" : 0.9, "pm" : 1./(2), "max_iter":1000,"selection":"proportional"}
        },
        "PS_E_REAL":
        {
            "Rastrigin":{"n_individuals" : 500, "pc" : 0.8, "pm" : 1./(2), "max_iter":1000, "elitism":0.1,"selection":"proportional"}, 
            "Beale":{"n_individuals" : 500, "pc" : 0.8, "pm" : 1/(2), "max_iter":1000, "elitism":0.1,"selection":"proportional"},
            "Himmelblau":{"n_individuals" : 500, "pc" : 0.8, "pm" : 1/(2), "max_iter":1000, "elitism":0.1,"selection":"proportional"}, 
            "Eggholder":{"n_individuals" : 500, "pc" : 0.8, "pm" : 1/(2), "max_iter":1000, "elitism":0.2,"selection":"proportional"}
        },
        "TS_REAL":
        {
            "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : 1/(2),"max_iter":1000,"selection":"tournament"}, 
            "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1/(2),"max_iter":1000,"selection":"tournament"},
            "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" :1/(2),"max_iter":1000,"selection":"tournament"}, 
            "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1/(2),"max_iter":1000,"selection":"tournament"}
        },
        "TS_E_REAL":
        {
            "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : 1/(2),"max_iter":1000,"selection":"tournament","elitism":0.1}, 
            "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1/2,"max_iter":1000,"selection":"tournament","elitism":0.1},
            "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1/2,"max_iter":1000,"selection":"tournament","elitism":0.1}, 
            "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./2,"max_iter":1000,"selection":"tournament","elitism":0.2}
        },
        "SUS_REAL":
        {
            "Rastrigin":{"n_individuals" : 500,"pc" : 0.9,"pm" :1/2,"max_iter":1000,"selection":"sus"}, 
            "Beale":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./2,"max_iter":1000,"selection":"sus"},
            "Himmelblau":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./2,"max_iter":1000,"selection":"sus"}, 
            "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./2,"max_iter":1000,"selection":"sus"}
        },
        "SUS_E_REAL":
        {
            "Rastrigin":{"n_individuals" : 500,"pc" : 0.9,"pm" : 1/2,"max_iter":1000,"selection":"sus","elitism":0.1}, 
            "Beale":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./2,"max_iter":1000,"selection":"sus","elitism":0.1},
            "Himmelblau":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./2,"max_iter":1000,"selection":"sus","elitism":0.1}, 
            "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./2,"max_iter":1000,"selection":"sus","elitism":0.2}
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
    start(fns, params, avg_times, avg_iters, "REAL")


