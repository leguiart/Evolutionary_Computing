from hklearn_genetic.genetic_algorithm import GeneticAlgorithm
from hklearn_genetic.problem import Rastrigin, Beale, Himmelblau, Eggholder
from scipy import signal
from utils import average_list_of_lists
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import logging.handlers
import os



rast = Rastrigin(n_dim = 2, n_prec=8)
beale = Beale(n_prec=8)
himme = Himmelblau(n_prec=8)
egg = Eggholder(n_prec=4)

fns = {"Rastrigin" : rast, "Beale" : beale, "Himmelblau" : himme, "Eggholder" : egg}
params = {
    "PS":
    {
        "Rastrigin":{"n_individuals" : 500, "pc" : 0.95, "pm" : 0.5/(rast.gene_length*2), "max_iter":10000,"selection":"proportional"}, 
        "Beale":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(beale.gene_length*2), "max_iter":10000,"selection":"proportional"},
        "Himmelblau":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(himme.gene_length*2), "max_iter":10000,"selection":"proportional"}, 
        "Eggholder":{"n_individuals" : 500, "pc" : 0.95, "pm" : 0.25/(egg.gene_length*2), "max_iter":10000,"selection":"proportional"}
    },
    "PS_E":
    {
        "Rastrigin":{"n_individuals" : 500, "pc" : 0.9, "pm" : .5/(rast.gene_length*2), "max_iter":10000, "elitism":0.1,"selection":"proportional"}, 
        "Beale":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(beale.gene_length*2), "max_iter":10000, "elitism":0.1,"selection":"proportional"},
        "Himmelblau":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(himme.gene_length*2), "max_iter":10000, "elitism":0.1,"selection":"proportional"}, 
        "Eggholder":{"n_individuals" : 500, "pc" : 0.95, "pm" : 1./(egg.gene_length*2), "max_iter":10000, "elitism":0.2,"selection":"proportional"}
    },
    "TS":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : .5/(rast.gene_length*2),"max_iter":10000,"selection":"tournament"}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : .25/(beale.gene_length*2),"max_iter":10000,"selection":"tournament"},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : 0.25/(himme.gene_length*2),"max_iter":10000,"selection":"tournament"}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : .25/(egg.gene_length*2),"max_iter":10000,"selection":"tournament"}
    },
    "TS_E":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : .5/(rast.gene_length*2),"max_iter":10000,"selection":"tournament","elitism":0.1}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : 25./(beale.gene_length*2),"max_iter":10000,"selection":"tournament","elitism":0.1},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : .25/(himme.gene_length*2),"max_iter":10000,"selection":"tournament","elitism":0.1}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":10000,"selection":"tournament","elitism":0.2}
    },
    "SUS":
    {
        "Rastrigin":{"n_individuals" : 500,"pc" : 0.9,"pm" :.5/(rast.gene_length*2),"max_iter":10000,"selection":"sus"}, 
        "Beale":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(beale.gene_length*2),"max_iter":10000,"selection":"sus"},
        "Himmelblau":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":10000,"selection":"sus"}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":10000,"selection":"sus"}
    },
    "SUS_E":
    {
        "Rastrigin":{"n_individuals" : 500,"pc" : 0.9,"pm" : .5/(rast.gene_length*2),"max_iter":10000,"selection":"sus","elitism":0.1}, 
        "Beale":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(beale.gene_length*2),"max_iter":10000,"selection":"sus","elitism":0.1},
        "Himmelblau":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":10000,"selection":"sus","elitism":0.1}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":10000,"selection":"sus","elitism":0.2}
    }
}

 
handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "tests.log"))
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
        gens_avg = 0
        ga_best_li = []
        ga_avg_li = []
        for i in range(5):
            if param[-1] == "E":
                #elitismo
                ga = GeneticAlgorithm(pc = params[param][p]["pc"], pm = params[param][p]["pm"], max_iter=params[param][p]["max_iter"], elitism=params[param][p]["elitism"], selection=params[param][p]["selection"])
            else:
                #sin elitismo
                ga = GeneticAlgorithm(pc = params[param][p]["pc"], pm = params[param][p]["pm"], max_iter=params[param][p]["max_iter"], selection=params[param][p]["selection"])

            print(f"Run: "f"{i}")
            logging.info(f"Run: "f"{i}")
            sol, gens = ga.evolve(fns[p], params[param][p]["n_individuals"])
            if gens < params[param][p]["max_iter"]:
                logging.info(f"Global minimum found: "f"{sol}")
                logging.info(f"after: "f"{gens}"f" generations.")
            else:
                logging.info(f"Local minimum found: "f"{sol}")
                logging.info(f"after maximum number of generations ("f"{gens}"f") elapsed.")
            gens_avg += gens/5.
            ga_best_li += [ga.best]
            ga_avg_li += [ga.averages]
            print(f"Solution(s) found: "f"{sol}")
            logging.info(f"Solution(s) found: "f"{sol}")
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
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        for i, data in enumerate(ga_best_li):
            if len(data) <= int(gens_avg):
                axs[0].plot(np.arange(len(data)), data, color = colors[i] ,label = f'Run: {i}')
            else:
                axs[0].plot(np.arange(int(gens_avg)), data[0 : int(gens_avg)], color = colors[i], label = f'Run: {i}')
        for i, data in enumerate(ga_avg_li):
            if len(data) <= int(gens_avg):
                axs[1].plot(np.arange(len(data)), data, color = colors[i], label = f'Run: {i}')   
            else:
                axs[1].plot(np.arange(int(gens_avg)), data[0 : int(gens_avg)], color = colors[i], label = f'Run: {i}')
        ga_best_avg = average_list_of_lists(ga_best_li)
        ga_avg_avg = average_list_of_lists(ga_avg_li)
        axs[0].plot(np.arange(int(gens_avg)), ga_best_avg[0:int(gens_avg)], color = 'r', label = 'Average')
        axs[0].set_title(p)
        axs[0].set_xlabel('Generations')
        axs[0].set_ylabel('Best fitness')
        fig.suptitle(f'{param}, 'f'5 run average', fontsize=16) 
        axs[1].plot(np.arange(int(gens_avg)), ga_avg_avg[0:int(gens_avg)], color = 'r', label = 'Average')
        axs[1].set_title(p)
        axs[1].set_xlabel('Generations')
        axs[1].set_ylabel('Average fitness') 
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()   
        plt.legend()
        #plt.show()
        plt.savefig(f"{param}"f"_"f"{p}"f"_"f"avg", bbox_inches='tight')
