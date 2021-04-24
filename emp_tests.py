from hklearn_genetic.genetic_algorithm import GeneticAlgorithm
from hklearn_genetic.problem import Rastrigin, Beale, Himmelblau, Eggholder
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import logging.handlers
import os



rast = Rastrigin(n_dim = 2)
beale = Beale()
himme = Himmelblau()
egg = Eggholder()

fns = {"Rastrigin" : rast, "Beale" : beale, "Himmelblau" : himme, "Eggholder" : egg}
params = {
    "PS":
    {
        "Rastrigin":{"n_individuals" : 250, "pc" : 0.95, "pm" : 0.5/(rast.gene_length*2), "max_iter":10000,"selection":"proportional"}, 
        "Beale":{"n_individuals" : 250, "pc" : 0.95, "pm" : .25/(beale.gene_length*2), "max_iter":10000,"selection":"proportional"},
        "Himmelblau":{"n_individuals" : 500, "pc" : 0.95, "pm" : .25/(himme.gene_length*2), "max_iter":10000,"selection":"proportional"}, 
        "Eggholder":{"n_individuals" : 500, "pc" : 0.95, "pm" : .5/(egg.gene_length*2), "max_iter":10000,"selection":"proportional"}
    },
    "PS_E":
    {
        "Rastrigin":{"n_individuals" : 100, "pc" : 0.9, "pm" : .1/(rast.gene_length*2), "max_iter":50000, "elitism":0.1,"selection":"proportional"}, 
        "Beale":{"n_individuals" : 100, "pc" : 0.95, "pm" : .1/(beale.gene_length*2), "max_iter":50000, "elitism":0.1,"selection":"proportional"},
        "Himmelblau":{"n_individuals" : 100, "pc" : 0.95, "pm" : .1/(himme.gene_length*2), "max_iter":50000, "elitism":0.1,"selection":"proportional"}, 
        "Eggholder":{"n_individuals" : 500, "pc" : 0.95, "pm" : 3./(egg.gene_length*2), "max_iter":100000, "elitism":0.1,"selection":"proportional"}
    },
    "TS":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : 1./(rast.gene_length*2),"max_iter":50000,"selection":"tournament"}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : .1/(beale.gene_length*2),"max_iter":50000,"selection":"tournament"},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":50000,"selection":"tournament"}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1.2/(egg.gene_length*2),"max_iter":100000,"selection":"tournament"}
    },
    "TS_E":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : 1./(rast.gene_length*2),"max_iter":50000,"selection":"tournament","elitism":0.1}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1./(beale.gene_length*2),"max_iter":50000,"selection":"tournament","elitism":0.1},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":50000,"selection":"tournament","elitism":0.1}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":100000,"selection":"tournament","elitism":0.1}
    },
    "SUS":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : 0.1,"max_iter":50000,"selection":"sus"}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1./(beale.gene_length*2),"max_iter":50000,"selection":"sus"},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":50000,"selection":"sus"}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":100000,"selection":"sus"}
    },
    "SUS_E":
    {
        "Rastrigin":{"n_individuals" : 100,"pc" : 0.9,"pm" : 0.1,"max_iter":50000,"selection":"sus","elitism":0.1}, 
        "Beale":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1./(beale.gene_length*2),"max_iter":50000,"selection":"sus","elitism":0.1},
        "Himmelblau":{"n_individuals" : 100,"pc" : 0.95,"pm" : 1./(himme.gene_length*2),"max_iter":50000,"selection":"sus","elitism":0.1}, 
        "Eggholder":{"n_individuals" : 500,"pc" : 0.95,"pm" : 1./(egg.gene_length*2),"max_iter":100000,"selection":"sus","elitism":0.1}
    }
}

 
handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "tests.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

print(matplotlib.get_backend())
for i in range(1):
    for param in params.keys():
        for p in params[param].keys():
            if param[-1] == "E":
                #elitismo
                ga = GeneticAlgorithm(pc = params[param][p]["pc"], pm = params[param][p]["pm"], max_iter=params[param][p]["max_iter"], elitism=params[param][p]["elitism"], selection=params[param][p]["selection"])
            else:
                #sin elitismo
                ga = GeneticAlgorithm(pc = params[param][p]["pc"], pm = params[param][p]["pm"], max_iter=params[param][p]["max_iter"], selection=params[param][p]["selection"])
            logging.info(f"Run: "f"{i}")
            logging.info(f"Selection method: "f"{param}")
            logging.info(f"Function to optimize: "f"{p}")
            logging.info(f"Parameters: "f"{params[param][p]}")
            print(f"Run: "f"{i}")
            print(f"Selection method: "f"{param}")
            print(f"Function to optimize: "f"{p}")
            print(f"Parameters: "f"{params[param][p]}")
            sol, gens = ga.evolve(fns[p], params[param][p]["n_individuals"])
            logging.info(f"Global minimum found: "f"{sol}")
            logging.info(f"after: "f"{gens}"f" generations.")
            print(f"Global minimum found: "f"{sol}")
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            axs[0].plot(np.arange(gens), ga.best)
            axs[0].set_title(p)
            axs[0].set_xlabel('Generations')
            axs[0].set_ylabel('Best fitness')
            fig.suptitle(param, fontsize=16) 
            axs[1].plot(np.arange(gens), ga.averages)
            axs[1].set_title(p)
            axs[1].set_xlabel('Generations')
            axs[1].set_ylabel('Average fitness') 
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()   
            #plt.savefig(f"{param}"f"_"f"{p}"f"_"f"{i}", bbox_inches='tight')
            plt.show()
