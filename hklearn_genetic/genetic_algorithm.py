
from abc import abstractmethod
import numpy as np
import random
import math

class GeneticAlgorithm:
    def __init__(self, pc = 0.6, pm = 0.1, max_iter = 5000, elitism = None, selection = "proportional"):
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm
        #print(pm)
        self.elitism = elitism
        self.selection = selection
        self.best = []
        self.averages = []
        
    
    def evolve(self, problem, n_individuals):
        its = 1
        self.pop = problem.populate(n_individuals)
        while its <= self.max_iter:
            self.pop, solution = self.select(problem, self.pop)
            if len(solution) > 0:
                print(its)
                if type(self.pop) is np.ndarray:
                    return problem.decode(self.pop[solution, :]), its
                else:
                    sol = []
                    for s in solution:
                        sol += [self.pop[s]]
                    return sol, its
            self.pop = problem.crossover(self.pop, self.pc, self.elitism)
            self.pop = problem.mutate(self.pop, self.pm, self.elitism)
            its+=1
        eval_X = problem.evaluate(self.pop)

        eval_X = np.sort(eval_X, order = ['fitness'])['index']
        if type(self.pop) is np.ndarray:
            return problem.decode(self.pop[eval_X[0:2], :]), its-1
        sol = []
        for i in range(2):
            sol += [self.pop[eval_X[i]]]
        return sol, its - 1


    def get_roulette_probs(self, n_individuals):
        return np.random.uniform(size = (1, n_individuals))

    def select(self, problem, X):
        eval_X = problem.evaluate(X)
        #print(eval_X)
        solution = problem.stop_criteria(eval_X['fitness'])
        self.best += [np.sort(eval_X, order = ['fitness'])['fitness'][len(eval_X) - 1]]
        avg_fitness = eval_X['fitness'].sum()/len(eval_X)
        self.averages += [avg_fitness]
        if len(solution) > 0:
            return X, solution
        
        if not self.elitism:
            if self.selection == "proportional":
                #eval_X = np.sort(eval_X, order = ['fitness'])
                #print(eval_X)
                prob_sel = eval_X['fitness']/eval_X['fitness'].sum()
                #print(prob_sel)
                c_prob_sel = np.cumsum(prob_sel)
                probs = self.get_roulette_probs(X.shape[0])
                
                # print(probs)
                # print(prob_sel)
                # print(c_prob_sel)
                new_X = np.zeros(X.shape, dtype = X.dtype)
                for j, prob in enumerate(probs[0, :]):
                    i = np.searchsorted(c_prob_sel, prob)
                    # print(i)
                    # print(X[i, :])
                    new_X[j, :] = X[i, :]
                #return new_X, solution
            elif self.selection == "tournament":
                if type(X) is np.ndarray:
                    new_X = np.zeros(X.shape, dtype = X.dtype)
                    # print(eval_X)
                    for i in range(2):
                        perm = np.random.permutation(X.shape[0])
                        # print(perm)
                        for j in range(0, X.shape[0]//2):
                            index = [perm[2 * j], perm[2 * j + 1]]
                            max_ind = np.argmax([eval_X['fitness'][index[0]], eval_X['fitness'][index[1]]]) 
                            # print(max_ind)
                            new_X[i * X.shape[0]//2 + j, :] = X[index[max_ind], :]
                elif type(X) is list:
                    new_X = []
                    for i in range(2):
                        perm = np.random.permutation(len(X))
                        # print(perm)
                        for j in range(0, len(X)//2):
                            index = [perm[2 * j], perm[2 * j + 1]]
                            max_ind = np.argmax([eval_X['fitness'][index[0]], eval_X['fitness'][index[1]]]) 
                            # print(max_ind)
                            new_X += [X[index[max_ind]]]
                #return new_X, solution
            elif self.selection == "sus":
                start = random.random()
                # print(start)
                pointers = [start*avg_fitness + i*avg_fitness for i in range(X.shape[0])]
                # print(pointers)
                new_X = np.zeros(X.shape, dtype = X.dtype)
                cum_fitness = np.cumsum(eval_X['fitness'])
                for j, p in enumerate(pointers):
                    i = np.searchsorted(cum_fitness, p)
                    # print(i)
                    new_X[j, :] = X[i, :]
                #return new_X, solution


        else:
            elitism_num = math.floor(self.elitism * len(eval_X))
            if self.selection == "proportional":
                eval_X = np.sort(eval_X, order = ['fitness'])
                # print(eval_X)
                
                #prob_sel = eval_X['fitness'][0:elitism_num]/eval_X['fitness'][0:elitism_num].sum()
                prob_sel = eval_X['fitness']/eval_X['fitness'].sum()
                c_prob_sel = np.cumsum(prob_sel)
                probs = self.get_roulette_probs(len(eval_X) - elitism_num)
                # print(probs)
                # print(prob_sel)
                # print(c_prob_sel)
                if type(X) is np.ndarray:
                    new_X = np.zeros(X.shape, dtype = X.dtype)
                    new_X[0 : elitism_num, :] = X[eval_X['index'][eval_X.size - elitism_num: eval_X.size], :]
                    #eval_X = eval_X[eval_X.size - elitism_num: eval_X.size]
                    # print(eval_X)
                    for j, prob in enumerate(probs[0, :]):
                        i = np.searchsorted(c_prob_sel, prob)
                        # print(i)
                        # print(X[eval_X['index'][i], :])
                        new_X[j + elitism_num, :] = X[eval_X['index'][i], :]
                elif isinstance(X, list):
                    new_X = []
                    for j in range(elitism_num):
                        new_X[j]+=[X[eval_X['index'][eval_X.size - elitism_num + j]]]
                    for j, prob in enumerate(probs[0, :]):
                        i = np.searchsorted(c_prob_sel, prob)
                        new_X += [X[eval_X['index'][i]]]

                #return new_X, solution
            elif self.selection == "tournament":
                if type(X) is np.ndarray:
                    new_X = np.zeros(X.shape, dtype = X.dtype)
                    new_X[0 : elitism_num, :] = X[eval_X['index'][eval_X.size - elitism_num: eval_X.size], :]
                    # print(eval_X)
                    for i in range(2):
                        perm = np.random.permutation(X.shape[0])
                        # print(perm)
                        for j in range(0, (X.shape[0] - elitism_num)//2):
                            index = [perm[2 * j], perm[2 * j + 1]]
                            max_ind = np.argmax([eval_X['fitness'][index[0]], eval_X['fitness'][index[1]]]) 
                            # print(max_ind)
                            new_X[i * (X.shape[0] - elitism_num)//2 + j + elitism_num, :] = X[index[max_ind], :]
                else:
                    new_X = []
                    for j in range(elitism_num):
                        new_X += [X[eval_X['index'][eval_X.size - elitism_num + j]]]
                    for i in range(2):
                        perm = np.random.permutation(len(X))
                        # print(perm)
                        for j in range(0, (len(X) - elitism_num)//2):
                            index = [perm[2 * j], perm[2 * j + 1]]
                            max_ind = np.argmax([eval_X['fitness'][index[0]], eval_X['fitness'][index[1]]]) 
                            # print(max_ind)
                            new_X += [X[index[max_ind]]]
                #return new_X, solution
            elif self.selection == "sus":
                prob = eval_X['fitness'].sum()/(X.shape[0] - elitism_num)
                start = random.random()
                pointers = [start*prob + i*prob for i in range(X.shape[0] - elitism_num)]
                # print(pointers)
                new_X = np.zeros(X.shape, dtype = X.dtype)
                new_X[0 : elitism_num, :] = X[eval_X['index'][eval_X.size - elitism_num: eval_X.size], :]
                cum_fitness = np.cumsum(eval_X['fitness'])
                for j, p in enumerate(pointers):
                    i = np.searchsorted(cum_fitness, p)
                    # print(i)
                    new_X[j + elitism_num, :] = X[i, :]
        return new_X, solution




    
