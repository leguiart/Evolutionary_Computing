from hklearn_genetic.genetic_algorithm import GeneticAlgorithm
from hklearn_genetic.problem import IntegerNQueen, RealNQueen, BinaryNQueen
from scipy import signal
from utils import average_list_of_lists
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import logging.handlers
import os


# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
# child_means = [25, 32, 34, 20, 25]

# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width, men_means, width, label='Men')
# rects2 = ax.bar(x, women_means, width, label='Women')
# rects3 = ax.bar(x + width, child_means, width, label='Children')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

# fig.tight_layout()

# plt.show()

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "nqueen_tests.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)


colors = ['b', 'g', 'y', 'c', 'k', 'm']
params = {"Integer coding":500, "Real coding":500, "Binary coding":1000}
integer_means = []
integer_best = []
integer_global_count = []
real_means = []
real_best = []
real_global_count = []
binary_means = []
binary_best = []
binary_global_count = []
n_groups = list(range(12, 12 + 6*4 - 1, 4))
for dim in n_groups:
    gens_li_dict = {"Integer coding":[], "Real coding":[], "Binary coding":[]}
    ga_best_li_dict = {"Integer coding":[], "Real coding":[], "Binary coding":[]}
    ga_avg_li_dict = {"Integer coding":[], "Real coding":[], "Binary coding":[]}
    integer_best += [float('inf')]
    real_best += [float('inf')]
    binary_best += [float('inf')]
    integer_global_count += [0]
    real_global_count += [0]
    binary_global_count += [0]
    logging.info(f"Solutions for N = "f"{dim}")
    print(f"Solutions for N = "f"{dim}")
    for run in range(5):
        i_nqueen = IntegerNQueen(n_dim = dim)
        r_nqueen = RealNQueen(n_dim=dim)
        b_nqueen = BinaryNQueen(n_dim=dim)
        logging.info(f"Solutions for run = "f"{run}")
        print(f"Solutions for run = "f"{run}")

        ga = GeneticAlgorithm(pc = 0.9, pm = 1./dim, max_iter= params["Integer coding"], selection="tournament", elitism=0.1)
        sol, gens = ga.evolve(i_nqueen, 500)
        if gens < params["Integer coding"]:
            integer_best[-1] = 0
            integer_global_count[-1]+=1
            logging.info("Integer coding global solution: ")
            print("Integer coding global solution: ")
        else:
            integer_best[-1] = -ga.best[-1] if integer_best[-1] > -ga.best[-1] else integer_best[-1]
            logging.info("Integer coding local solution: ")
            print("Integer coding local solution: ")
        logging.info(sol)
        print(sol)
        logging.info(f"After "f"{gens}"f" generations")
        
        best = [-b for b in ga.best]
        avgs = [-avg for avg in ga.averages]
        gens_li_dict["Integer coding"]+=[gens]
        ga_best_li_dict["Integer coding"]+=[best]
        ga_avg_li_dict["Integer coding"]+=[avgs]
        # fig, axs = plt.subplots(2, 1, constrained_layout=True)
        # axs[0].plot(np.arange(gens), best)
        # axs[0].set_title(f"Integer coding, N = "f"{dim}")
        # axs[0].set_xlabel('Generations')
        # axs[0].set_ylabel('Best fitness')
        # fig.suptitle("N Queen Problem", fontsize=16) 
        # axs[1].plot(np.arange(gens), avgs)
        # axs[1].set_title(f"Integer coding, N = "f"{dim}")
        # axs[1].set_xlabel('Generations')
        # axs[1].set_ylabel('Average fitness') 
        # plt.show()

        ga = GeneticAlgorithm(pc = 0.9, pm = 1./dim, max_iter= params["Real coding"], selection="tournament", elitism=0.1)
        sol, gens = ga.evolve(r_nqueen, 500)
        if gens < params["Real coding"]:
            real_global_count[-1]+=1
            real_best[-1] = 0
            logging.info("Real coding global solution: ")
            print("Real coding global solution: ")
        else:
            real_best[-1] = -ga.best[-1] if real_best[-1] > -ga.best[-1] else real_best[-1]
            logging.info("Real coding local solution: ")
            print("Real coding local solution: ")
        logging.info(sol)
        print(sol)
        logging.info(f"After "f"{gens}"f" generations")
        best = [-b for b in ga.best]
        avgs = [-avg for avg in ga.averages]
        gens_li_dict["Real coding"]+=[gens]
        ga_best_li_dict["Real coding"]+=[best]
        ga_avg_li_dict["Real coding"]+=[avgs]
        # fig, axs = plt.subplots(2, 1, constrained_layout=True)
        # axs[0].plot(np.arange(gens), best)
        # axs[0].set_title(f"Real coding, N = "f"{dim}")
        # axs[0].set_xlabel('Generations')
        # axs[0].set_ylabel('Best fitness')
        # fig.suptitle("N Queen Problem", fontsize=16) 
        # axs[1].plot(np.arange(gens), avgs)
        # axs[1].set_title(f"Real coding, N = "f"{dim}")
        # axs[1].set_xlabel('Generations')
        # axs[1].set_ylabel('Average fitness') 
        # plt.show()

        ga = GeneticAlgorithm(pc = 0.9, pm = 1./(b_nqueen.gene_length*b_nqueen.n_dim), max_iter= params["Binary coding"], selection="tournament", elitism=0.1)
        sol, gens = ga.evolve(b_nqueen, 500)
        if gens < params["Binary coding"]:
            binary_global_count[-1]+=1
            binary_best[-1] = 0
            logging.info("Binary coding global solution: ")
            print("Binary coding global solution: ")
        else:
            binary_best[-1] = -ga.best[-1] if binary_best[-1] > -ga.best[-1] else binary_best[-1]
            logging.info("Binary coding local solution: ")
            print("Binary coding local solution: ")
        logging.info(sol)
        print(sol)
        logging.info(f"After "f"{gens}"f" generations")
        best = [-b for b in ga.best]
        avgs = [-avg for avg in ga.averages]
        gens_li_dict["Binary coding"]+=[gens]
        ga_best_li_dict["Binary coding"]+=[best]
        ga_avg_li_dict["Binary coding"]+=[avgs]
        # fig, axs = plt.subplots(2, 1, constrained_layout=True)
        # axs[0].plot(np.arange(gens), best)
        # axs[0].set_title(f"Binary coding, N = "f"{dim}")
        # axs[0].set_xlabel('Generations')
        # axs[0].set_ylabel('Best fitness')
        # fig.suptitle("N Queen Problem", fontsize=16) 
        # axs[1].plot(np.arange(gens), avgs)
        # axs[1].set_title(f"Binary coding, N = "f"{dim}")
        # axs[1].set_xlabel('Generations')
        # axs[1].set_ylabel('Average fitness') 
        # plt.show()

    for key in ["Integer coding", "Real coding", "Binary coding"]:
        gens_avg = np.average(gens_li_dict[key])
        if key == "Integer coding":
            integer_means += [gens_avg]
        elif key == "Real coding":
            real_means += [gens_avg]
        else:
            binary_means += [gens_avg]
        # for i in range(len(ga_best_li_dict[key])):
        #     if len(ga_best_li_dict[key][i]) < params[key]:
        #         ga_best_li_dict[key][i] += list(np.zeros(params[key] - len(ga_best_li_dict[key][i])))
        # for i in range(len(ga_avg_li_dict[key])):
        #     if len(ga_avg_li_dict[key][i]) < params[key]:
        #         ga_avg_li_dict[key][i] += list(np.zeros(params[key] - len(ga_avg_li_dict[key][i])))
        # ga_best_avg = np.average(np.array(ga_best_li_dict[key]), axis=0)
        # ga_avg_avg = np.average(np.array(ga_avg_li_dict[key]), axis=0)
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        for i, data in enumerate(ga_best_li_dict[key]):
            if len(data) <= int(gens_avg):
                axs[0].plot(np.arange(len(data)), data, color = colors[i] ,label = f'Run: {i}')
            else:
                axs[0].plot(np.arange(int(gens_avg)), data[0 : int(gens_avg)], color = colors[i], label = f'Run: {i}')
        for i, data in enumerate(ga_avg_li_dict[key]):
            if len(data) <= int(gens_avg):
                axs[1].plot(np.arange(len(data)), data, color = colors[i], label = f'Run: {i}')   
            else:
                axs[1].plot(np.arange(int(gens_avg)), data[0 : int(gens_avg)], color = colors[i], label = f'Run: {i}')
        ga_best_avg = average_list_of_lists(ga_best_li_dict[key])
        ga_avg_avg = average_list_of_lists(ga_avg_li_dict[key])
        
        axs[0].plot(np.arange(int(gens_avg)), ga_best_avg[0 : int(gens_avg)], color = 'r', label = 'Average')
        axs[0].set_title(f"{key} "f", N = "f"{dim}")
        axs[0].set_xlabel('Generations')
        axs[0].set_ylabel('Best fitness')
        fig.suptitle("N Queen Problem for 5 runs", fontsize=16) 
        axs[1].plot(np.arange(int(gens_avg)), ga_avg_avg[0 : int(gens_avg)], color = 'r', label = 'Average')
        axs[1].set_title(f"{key} "f", N = "f"{dim}")
        axs[1].set_xlabel('Generations')
        axs[1].set_ylabel('Average fitness') 
        plt.legend()
        plt.savefig(f"{key}"f"_{dim}", bbox_inches='tight')
        #plt.show()

x = np.arange(len(n_groups))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, integer_best, width, label='Integer')
rects2 = ax.bar(x, real_best, width, label='Real')
rects3 = ax.bar(x + width, binary_best, width, label='Binary')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Best fitness')
ax.set_title('Best fitness by Nqueens and coding')
ax.set_xticks(x)
ax.set_xticklabels(n_groups)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.savefig("Best_Fitness", bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, integer_global_count, width, label='Integer')
rects2 = ax.bar(x, real_global_count, width, label='Real')
rects3 = ax.bar(x + width, binary_global_count, width, label='Binary')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Global solutions')
ax.set_title('Global solutions found by Nqueens and coding')
ax.set_xticks(x)
ax.set_xticklabels(n_groups)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.savefig("Global_Solutions", bbox_inches='tight')
#plt.show()