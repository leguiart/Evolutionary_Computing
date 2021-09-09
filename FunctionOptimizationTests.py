from hklearn_genetic.utils import ProblemUtils
from hklearn_genetic.Evaluators.FunctionEvaluators import BaseBeale, BaseEggholder, BaseHimmelblau, BaseRastrigin, Sphere, Schaffer
from hklearn_genetic.Problems import RealBGAProblem, RealGAProblem, IntegerGAProblem, BinaryGAProblem
from hklearn_genetic.EvolutionaryAlgorithm import EvolutionaryAlgorithm
import numpy as np
import json
import os
import os.path

class MyAnalytics:
    def __init__(self, function_name):
        self.analytics = {"name" : function_name,"eval_mat" : [], "best_fitness" : [], "average_fitness":[]}

    def gather_analytics(self, X_eval):
        X_eval_li, X_eval_mat = ProblemUtils._to_evaluated_matrix(X_eval)
        self.analytics["eval_mat"]+=[X_eval_li]
        self.analytics["best_fitness"]+=[list(X_eval_mat[int(np.argmin(np.array(X_eval_mat[:, 2]), axis = 0)), :])]
        self.analytics["average_fitness"]+=[float(X_eval_mat[:, 2].mean())]


analytics_sphere = MyAnalytics("sphere")
analytics_schaffer = MyAnalytics("schaffer")
#Defining evaluators
beale = BaseBeale(rank=0)
eggholder = BaseEggholder(rank = 0)
himmelblau = BaseHimmelblau(rank = 0)
rastrigin = BaseRastrigin(rank=0)
sphere = Sphere()
schaffer = Schaffer()

#Defining problems
#Real SBX crossover and polynomial mutation
beale_real_sbx = RealGAProblem._BaseRealGAProblem(beale, -0.001, (-4.5, 4.5), pc = 0.85, pm = 0.1)
eggholder_real_sbx = RealGAProblem._BaseRealGAProblem(eggholder, 959.63, (-512., 512.), pc = 0.85, pm = 0.1)
himmelblau_real_sbx = RealGAProblem._BaseRealGAProblem(himmelblau, -0.001, (-5., 5.), pc = 0.85, pm = 0.1)
rastrigin_real_sbx = RealGAProblem._BaseRealGAProblem(rastrigin, -0.0001, (-5.12, 5.12), pc = 0.85, pm = 0.1)
sphere_real_sbx = RealGAProblem._BaseRealGAProblem(sphere, -0.0001, (-5., 5.), pc = 0.85, pm = 0.1)
schaffer_real_sbx = RealGAProblem._BaseRealGAProblem(schaffer, -0.001, (-500, 500), pc = 0.85, pm = 0.1)
#Real BGA based
beale_real_bga = RealBGAProblem._BaseRealBGAProblem(beale, -0.0001, (-4.5, 4.5), pc = 0.85, pm = 0.1)
eggholder_real_bga = RealBGAProblem._BaseRealBGAProblem(eggholder, 959.63, (-512., 512.), pc = 0.85, pm = 0.1)
himmelblau_real_bga = RealBGAProblem._BaseRealBGAProblem(himmelblau, -0.0001, (-5., 5.), pc = 0.85, pm = 0.1)
rastrigin_real_bga = RealBGAProblem._BaseRealBGAProblem(rastrigin, -0.0001, (-5.12, 5.12), pc = 0.85, pm = 0.1)
sphere_real_bga = RealBGAProblem._BaseRealBGAProblem(sphere, -0.0001, (-5., 5.), pc = 0.85, pm = 0.1)
schaffer_real_bga = RealBGAProblem._BaseRealBGAProblem(schaffer, -0.001, (-500, 500), pc = 0.85, pm = 0.1)
#Integer
# beale_real_int = IntegerGAProblem._BaseIntegerGAProblem(beale, -0.0001, (-4.5, 4.5), pc = 0.85, pm = 0.1)
# eggholder_real_int = IntegerGAProblem._BaseIntegerGAProblem(eggholder, -0.0001, (-512., 512.), pc = 0.85, pm = 0.1)
# himmelblau_real_int = IntegerGAProblem._BaseIntegerGAProblem(himmelblau, -0.0001, (-5., 5.), pc = 0.85, pm = 0.1)
# rastrigin_real_int = IntegerGAProblem._BaseIntegerGAProblem(rastrigin, -0.0001, (-5.12, 5.12), pc = 0.85, pm = 0.1)
# sphere_real_int = IntegerGAProblem._BaseIntegerGAProblem(sphere, -0.0001, (-5., 5.), pc = 0.85, pm = 0.1)
# schaffer_real_int = IntegerGAProblem._BaseIntegerGAProblem(schaffer, -0.001, (-500, 500), pc = 0.85, pm = 0.1)
#Binary
beale_real_bin = BinaryGAProblem._BaseBinaryGAProblem(beale, -0.001, (-4.5, 4.5), pc = 0.85, pm = 0.01, n_prec=4)
eggholder_real_bin = BinaryGAProblem._BaseBinaryGAProblem(eggholder, 959.63, (-512., 512.), pc = 0.85, pm = 0.01, n_prec=4)
himmelblau_real_bin = BinaryGAProblem._BaseBinaryGAProblem(himmelblau, -0.001, (-5., 5.), pc = 0.85, pm = 0.01, n_prec=4)
rastrigin_real_bin = BinaryGAProblem._BaseBinaryGAProblem(rastrigin, -0.001, (-5.12, 5.12), pc = 0.85, pm = 0.01, n_prec=4)
sphere_real_bin = BinaryGAProblem._BaseBinaryGAProblem(sphere, -0.001, (-5., 5.), pc = 0.85, pm = 0.01, n_prec=4)
schaffer_real_bin = BinaryGAProblem._BaseBinaryGAProblem(schaffer, -0.001, (-500, 500), pc = 0.85, pm = 0.01, n_prec=4)

#Defining the evolutionary algorithm
ea = EvolutionaryAlgorithm(selection="tournament", tournament_type=1)


# print("Real SBX: ")
# print(ea.evolve(beale_real_sbx, 100))
# print(ea.evolve(eggholder_real_sbx, 100))
# print(ea.evolve(himmelblau_real_sbx, 100))
# print(ea.evolve(rastrigin_real_sbx, 100))
# print(ea.evolve(sphere_real_sbx, 100))
# print(ea.evolve(schaffer_real_sbx, 100))

# print("Real BGA: ")
# print(ea.evolve(beale_real_bga, 100))
# print(ea.evolve(eggholder_real_bga, 100))
# print(ea.evolve(himmelblau_real_bga, 100))
# print(ea.evolve(rastrigin_real_bga, 100))
# print(ea.evolve(sphere_real_bga, 100))
# print(ea.evolve(schaffer_real_bga, 100))

# print("Integer: ")
# print(ea.evolve(beale_real_int, 100))
# print(ea.evolve(eggholder_real_int, 100))
# print(ea.evolve(himmelblau_real_int, 100))
# print(ea.evolve(rastrigin_real_int, 100))
# print(ea.evolve(sphere_real_int, 100))
# print(ea.evolve(schaffer_real_int, 100))

print("Binary: ")
print(ea.evolve(beale_real_bin, 100))
print(ea.evolve(eggholder_real_bin, 100))
print(ea.evolve(himmelblau_real_bin, 100))
print(ea.evolve(rastrigin_real_bin, 100))
print(ea.evolve(sphere_real_bin, 100))
print(ea.evolve(schaffer_real_bin, 100))

# Dump analytics in a json in order to extract insights from it offline
# if os.path.isfile('sphere_analytics.json'):
#     with open('sphere_analytics.json', 'a') as fp:           
#         fp.write('\n')
#         json.dump(analytics_sphere.analytics, fp)
# else:
#     with open('sphere_analytics.json', 'w') as fp: 
#         json.dump(analytics_sphere.analytics, fp)

# if os.path.isfile('schaffer_analytics.json'):
#     with open('schaffer_analytics.json', 'a') as fp:           
#         fp.write('\n')
#         json.dump(analytics_schaffer.analytics, fp)
# else:
#     with open('schaffer_analytics.json', 'w') as fp: 
#         json.dump(analytics_schaffer.analytics, fp)