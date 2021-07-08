from numpy.core.shape_base import _arrays_for_stack_dispatcher
from hklearn_genetic.genetic_algorithm import GeneticAlgorithm
from hklearn_genetic.problem import SymbolicRegressionProblem, BitParityCheck, NeutralityProblem
from utils import average_list_of_lists, protectedDiv, protectedSqrt, protectedExp, sin, cos, tan, plot_superposed
from deap import gp
import matplotlib.pyplot as plt
import numpy as np
import logging
import logging.handlers
import os
import operator
import math
import random

GLOBAL_CONF = {"Symbolic Regression":True, "Parity Check":False, "Parity Check without XOR":False}
GLOBAL_CONF_NEUTRALITY = {"Neutrality_node" :   False, "Neutrality_branch" :  False, "Neutrality_product" :  False}


def plot_estimated_solution(est_solutions, srp, sub_title, sup_title, show = True, store = False, store_title = "symbolic_regression_estimate"):
    func = gp.compile(est_solutions, srp.pset)
    y_estim = np.array([func(*point) for point in srp.points])
    labels = ['estimated y', 'real y']
    y_s = [y_estim, srp.real_values]
    plot_superposed(srp.points, y_s, plot_labels=labels, sub_title=sub_title, sup_title = sup_title, show=show, store=store, store_path = store_title)

def plot_estimated_solutions(est_solutions, srp, sub_title, sup_title, show = True, store = False, store_title = "symbolic_regression_estimate"):
    funcs = list(map(lambda est_solution : (gp.compile(est_solution[0], srp.pset), est_solution[1]) if type(est_solution) is tuple else gp.compile(est_solution, srp.pset), est_solutions))
    y_estims = list(map(lambda func : ([func[0](*point) for point in srp.points], func[1]) if type(func) is tuple else [func(point) for point in srp.points] , funcs))
    labels = [None]*(len(y_estims) - 1)
    y_best = None
    y_original = None
    labels += ['best estimated y'] + ['real y']
    if type(y_estims[0]) is tuple:
        y_original = (srp.real_values, 'r')
    else:
        y_original = srp.real_values
    y_estims += [y_original]
    plot_superposed(srp.points, y_estims, plot_labels=labels, sub_title=sub_title, sup_title = sup_title, show=show, store=store, store_path = store_title)

def plot_gene_counts(gene_counts, generations, sup_title, show = True, store = False, store_title = "gene_count"):
    labels = []
    for gene in gene_counts.keys():
        labels += [f"Gene: {gene}"]
    x = np.arange(0, generations + 1)
    plot_superposed(x, gene_counts.values(), x_label="Generations", y_label="Number of Genes", plot_labels=labels, sup_title = sup_title, show=show, store=store, store_path = store_title)


handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "gp_tests.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

colors = ['b', 'g', 'y', 'c', 'k', 'm']

#y = "3.4370    3.3868    3.3651    3.3709    3.4031    3.4598    3.5394    3.6395    3.7578    3.8917    4.0384    4.1947    4.3576    4.5237    4.6896    4.8518    5.0065    5.1502    5.2792    5.3895    5.4774    5.5389    5.5697    5.5654    5.5208    5.4299    5.2843    5.0717    4.7694    4.3212    3.1416    4.3212    4.7694    5.0717    5.2843    5.4299    5.5208    5.5654    5.5697    5.5389    5.4774    5.3895    5.2792    5.1502    5.0065    4.8518    4.6896    4.5237    4.3576    4.1947    4.0384    3.8917    3.7578    3.6395    3.5394    3.4598    3.4031    3.3709    3.3651    3.3868    3.4370    3.5165    3.6256    3.7645    3.9327    4.1298    4.3548    4.6066    4.8837    5.1843    5.5065".split()
y = []
f = open('datos_examen_2021II.txt', 'r')
data = [[]]
for l in f:
    y.append(float(l))
#y = list(map(lambda x : float(x),y))

### Primitivas para regresión simbólica
#Creamos un conjunto de primitivas llamado "main"
pset = gp.PrimitiveSet("main", 1)

#Agregamos no terminales, primer argumento es la función, segundo argumento es la aridad
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
#pset.addPrimitive(math.tan, 1)
pset.addPrimitive(protectedExp, 1)
pset.addPrimitive(protectedSqrt, 1)
pset.addPrimitive(operator.abs, 1)

#Agregamos terminales (constantes)
# for i in range(1, 2):
#     pset.addTerminal(float(i))
#pset.addTerminal(math.pi)

#Agregamos terminales (argumentos), al agregar nuestras primitivas con su aridad, se crearon
#2 argumentos por defecto, los renombramos como 'x' y 'y'
pset.renameArguments(ARG0="x")
pset.addEphemeralConstant('eff1', lambda: random.uniform(-2.5, 0))
srp = SymbolicRegressionProblem([(0., 5.)], pset, y, 17, stop_thresh=100.)
#srp = SymbolicRegressionProblem([(1., float(len(y)))], pset, y, 17, stop_thresh=100.)

### Primitivas para paridad par
S = [True, False, False, True, False, True, True, False, False, True, True, False, True, False, False, True]
pset_binary = gp.PrimitiveSet("main", 4)
pset_binary.addPrimitive(operator.and_, 2)
pset_binary.addPrimitive(operator.or_, 2)
pset_binary.addPrimitive(operator.xor, 2)
pset_binary.addPrimitive(operator.not_, 1)
pset_binary.renameArguments(ARG0="x")
pset_binary.renameArguments(ARG1="y")
pset_binary.renameArguments(ARG2="z")
pset_binary.renameArguments(ARG3="w")
bpc = BitParityCheck(pset_binary, S, 17)



pset_binary2 = gp.PrimitiveSet("main2", 4)
#Agregamos no terminales, primer argumento es la función, segundo argumento es la aridad
pset_binary2.addPrimitive(operator.and_, 2)
pset_binary2.addPrimitive(operator.or_, 2)
pset_binary2.addPrimitive(operator.not_, 1)
pset_binary2.renameArguments(ARG0="x")
pset_binary2.renameArguments(ARG1="y")
pset_binary2.renameArguments(ARG2="z")
pset_binary2.renameArguments(ARG3="w")
bpc2 = BitParityCheck(pset_binary2, S, 17)


p_set_neutrality = gp.PrimitiveSet("neutrality", 0)
p_set_neutrality.addPrimitive(operator.add, 2)
neutrality = NeutralityProblem(p_set_neutrality, 10, 50, [0 , 1, 4], mutation_type="Node")
neutrality2 = NeutralityProblem(p_set_neutrality, 10, 50, [0 , 1, 4])
p_set_neutrality2 = gp.PrimitiveSet("neutrality2", 0)
p_set_neutrality2.addPrimitive(operator.add, 2)
p_set_neutrality2.addPrimitive(operator.mul, 2)
neutrality3 = NeutralityProblem(p_set_neutrality2, 10, 50, [0 , 1, 4])


gens_li_dict = {"Symbolic Regression":[], "Parity Check":[], "Parity Check without XOR":[]}
ga_best_li_dict = {"Symbolic Regression":[], "Parity Check":[], "Parity Check without XOR":[]}
ga_avg_li_dict = {"Symbolic Regression":[], "Parity Check":[], "Parity Check without XOR":[]}
ga_avg_length_li_dict = {"Symbolic Regression":[], "Parity Check":[], "Parity Check without XOR":[]}
symbolic_regression_best = float('inf')
parity_check_best = float('inf')
parity_check_wo_xor_best = float('inf')
symbolic_regression_best_solution = None
parity_check_best_solution = None
parity_check_wo_xor_best_solution = None
symbolic_regression_solutions = []

for i in range(5):
    if GLOBAL_CONF["Symbolic Regression"]:
        try_flag = True
        while try_flag:
            try:
                ga = GeneticAlgorithm(pc = 0.9, pm = 0.1, selection="tournament", max_iter=500, elitism=0.1)
                sol, gens = ga.evolve(srp, 100)
                try_flag = False
            except Exception as e:
                print(e)
                try_flag = True
                srp.avg_lengths = []

        #plot_estimated_solution(sol[0], srp,  "Y estimated vs Y real", f"Model for run {i}", show=False, store = True, store_title=f"symbolic_regression_{i}")
        if symbolic_regression_best > abs(ga.best[-1]):
            if symbolic_regression_best_solution:
                symbolic_regression_solutions += [(symbolic_regression_best_solution[0], "0.8")] 
            symbolic_regression_best = abs(ga.best[-1])
            symbolic_regression_best_solution = (sol[0], "b")
        else:
            symbolic_regression_solutions += [(sol[0], "0.8")] 
        if gens < 500:
            logging.info("Symbolic Regression global solution: ")
            print("Symbolic Regression global solution: ")
        else:
            logging.info("Symbolic Regression local solution: ")
            print("Symbolic Regression local solution: ")
        logging.info(gp.PrimitiveTree(sol[0]))
        print(gp.PrimitiveTree(sol[0]))
        logging.info(f"After "f"{gens}"f" generations")
        best = [-b for b in ga.best]
        avgs = [-avg for avg in ga.averages]

        gens_li_dict["Symbolic Regression"]+=[gens]
        ga_best_li_dict["Symbolic Regression"]+=[best]
        ga_avg_li_dict["Symbolic Regression"]+=[avgs]
        ga_avg_length_li_dict["Symbolic Regression"]+=[srp.avg_lengths]

    if GLOBAL_CONF["Parity Check"]:
        ga = GeneticAlgorithm(pc = 0.9, pm = 0.001,  selection="tournament", max_iter=500, elitism=0.1)
        sol, gens = ga.evolve(bpc, 300)
        if parity_check_best > -ga.best[-1]:
            parity_check_best = -ga.best[-1]
            parity_check_best_solution = gp.PrimitiveTree(sol[0])
        if gens < 500:
            logging.info("Parity Check global solution: ")
            print("Parity Check global solution: ")
        else:
            logging.info("Parity Check local solution: ")
            print("Parity Check local solution: ")
        logging.info(gp.PrimitiveTree(sol[0]))
        print(gp.PrimitiveTree(sol[0]))
        logging.info(f"After "f"{gens}"f" generations")
        best = [-b for b in ga.best]
        avgs = [-avg for avg in ga.averages]

        gens_li_dict["Parity Check"]+=[gens]
        ga_best_li_dict["Parity Check"]+=[best]
        ga_avg_li_dict["Parity Check"]+=[avgs]
        ga_avg_length_li_dict["Parity Check"]+=[bpc.avg_lengths]

    if GLOBAL_CONF["Parity Check without XOR"]:
        ga = GeneticAlgorithm(pc = 0.9, pm = 0.0,  selection="tournament", max_iter=500, elitism=0.1)
        sol, gens = ga.evolve(bpc2, 300)
        if parity_check_wo_xor_best > -ga.best[-1]:
            parity_check_wo_xor_best = -ga.best[-1]
            parity_check_wo_xor_best_solution = gp.PrimitiveTree(sol[0])
        if gens < 500:
            logging.info("Parity Check without XOR global solution: ")
            print("Parity Check without XOR global solution: ")
        else:
            logging.info("Parity Check without XOR local solution: ")
            print("Parity Check without XOR local solution: ")
        logging.info(gp.PrimitiveTree(sol[0]))
        print(gp.PrimitiveTree(sol[0]))
        logging.info(f"After "f"{gens}"f" generations")
        best = [-b for b in ga.best]
        avgs = [-avg for avg in ga.averages]

        gens_li_dict["Parity Check without XOR"]+=[gens]
        ga_best_li_dict["Parity Check without XOR"]+=[best]
        ga_avg_li_dict["Parity Check without XOR"]+=[avgs]
        ga_avg_length_li_dict["Parity Check without XOR"]+=[bpc2.avg_lengths]
    
    srp.avg_lengths = []
    bpc.avg_lengths = []
    bpc2.avg_lengths = []

keys = []
for key in GLOBAL_CONF.keys():
    if GLOBAL_CONF[key]:
        keys+=[key]


for key in keys:
    gens_avg = np.average(gens_li_dict[key])

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
    ga_avg_length_avg = average_list_of_lists(ga_avg_length_li_dict[key])

    axs[0].plot(np.arange(int(gens_avg)), ga_best_avg[0 : int(gens_avg)], color = 'r', label = 'Average')
    axs[0].set_title(f"{key}")
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Best fitness')
    fig.suptitle(f"{key} for 5 runs", fontsize=16) 
    axs[1].plot(np.arange(int(gens_avg)), ga_avg_avg[0 : int(gens_avg)], color = 'r', label = 'Average')
    axs[1].set_title(f"{key} ")
    axs[1].set_xlabel('Generations')
    axs[1].set_ylabel('Average fitness') 
    plt.legend()
    plt.savefig(f"{key}", bbox_inches='tight')

    
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.plot(np.arange(int(gens_avg)), ga_avg_length_avg[0 : int(gens_avg)])
    axs.set_title(f"{key} Average complexity")
    axs.set_xlabel('Generations')
    axs.set_ylabel('Number of genes')
    fig.suptitle(f"Average complexity for {key}, for 5 runs", fontsize=16) 
    plt.savefig(f"{key}_complexity", bbox_inches='tight')


if GLOBAL_CONF["Symbolic Regression"]:
    symbolic_regression_solutions += [symbolic_regression_best_solution]
    plot_estimated_solutions(symbolic_regression_solutions, srp, "Y estimated vs Y real", "Global solutions of the 5 runs", store=True, store_title="symbolic_regression_globals")
    plot_estimated_solution(gp.PrimitiveTree(symbolic_regression_best_solution[0]), srp, "Y estimated vs Y real", "Best model of the 5 runs", store=True, store_title="symbolic_regression_best")
    print(f"Symbolic regression best solution: \n {gp.PrimitiveTree(symbolic_regression_best_solution[0])}")
    logging.info(f"Symbolic regression best solution: \n {gp.PrimitiveTree(symbolic_regression_best_solution[0])}")

if GLOBAL_CONF_NEUTRALITY["Neutrality_node"]:
    ga = GeneticAlgorithm(pc = 0.9, pm = 0.001,  selection="tournament", max_iter=1000)
    sol, gens = ga.evolve(neutrality, 500)
    plot_gene_counts(neutrality.gene_counts, gens, sup_title="Gene count node mutation", show=True, store=True, store_title="gene_count_node")

if GLOBAL_CONF_NEUTRALITY["Neutrality_branch"]:
    ga = GeneticAlgorithm(pc = 0.9, pm = 0.01,  selection="tournament", max_iter=1000)
    sol, gens = ga.evolve(neutrality2, 500)
    plot_gene_counts(neutrality2.gene_counts, gens, sup_title="Gene count branch mutation", show=True, store=True, store_title="gene_count_branch")

if GLOBAL_CONF_NEUTRALITY["Neutrality_product"]:
    ga = GeneticAlgorithm(pc = 0.9, pm = 0.01,  selection="tournament", max_iter=1000)
    sol, gens = ga.evolve(neutrality3, 500)
    plot_gene_counts(neutrality3.gene_counts, gens, sup_title="Gene count branch mutation with product", show=True, store=True, store_title="gene_count_product")