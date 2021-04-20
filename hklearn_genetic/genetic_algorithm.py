
from abc import abstractmethod
import numpy as np
import random
import math

class GeneticAlgorithm:
    def __init__(self, n_individuals = 10, pc = 0.6, pm = 0.1, max_iter = 5000, elitism = None, selection = "proportional"):
        self.max_iter = max_iter
        self.n_individuals = n_individuals
        self.pc = pc
        self.pm = pm
        #print(pm)
        self.elitism = elitism
        self.selection = selection
        self.best = []
        
    
    def evolve(self, problem):
        its = 1
        self.pop = problem.populate(self.n_individuals)
        while its <= self.max_iter:
            self.pop, solution = self.select(problem, self.pop)
            if len(solution) > 0:
                print(its)
                return problem.decode(self.pop[solution, :]), its
            self.pop = self.crossover(self.pop)
            self.pop = self.mutate(self.pop)
            its+=1
        eval_X = problem.evaluate(self.pop)

        eval_X = np.sort(eval_X, order = ['fitness'])['index']
        return problem.decode(self.pop[eval_X[0:2], :]), its-1


    def mutate(self, X):
        mutate_m = np.random.rand(X.shape[0], X.shape[1])
        # print(mutate_m)
        # print(self.pm)
        mutate_m = mutate_m <= self.pm
        # print(mutate_m)
        X_bit = X == 1
        # print(X_bit)
        #elitism_num = 0 if not self.elitism else math.floor(self.elitism * X.shape[0])
        if not self.elitism:
            X = np.logical_xor(X_bit, mutate_m)
        else:
            elitism_num = math.floor(self.elitism * X.shape[0])
            X[X.shape[0] - elitism_num :  X.shape[0], :] = np.logical_xor(X_bit, mutate_m)[X.shape[0] - elitism_num :  X.shape[0], :]
        X = X.astype(int)
        return X


    def crossover(self, X):
        if not self.elitism:
            n_cross = self.n_individuals // 2
            elitism_num = 0
        else:
            elitism_num = math.floor(self.elitism * X.shape[0])
            n_cross = (self.n_individuals - elitism_num) // 2
        prob_cross = np.random.rand(1 , n_cross)[0,:]
        for i, p in enumerate(prob_cross):
            if p <= self.pc:
                cross_point = random.randint(1, X.shape[1])
                # print(i)
                # print(cross_point)
                son1 = X[2*i + elitism_num,:].copy()
                son2 = X[2*i + 1 + elitism_num, :].copy()
                son1[cross_point : -1] = X[2*i + 1 + elitism_num, cross_point : -1]
                son2[cross_point : -1] = X[2*i + elitism_num, cross_point : -1]
                # print(son1)
                # print(son2)
                X[2*i + elitism_num,:] = son1
                X[2*i + 1 + elitism_num,:] = son2
        return X


    def select(self, problem, X):
        eval_X = problem.evaluate(X)
        solution = problem.stop_criteria(eval_X['fitness'])
        self.best += [np.sort(eval_X, order = ['fitness'])['fitness'][X.shape[0] - 1]]
        if len(solution) > 0:
            return X, solution
        
        if not self.elitism:
            if self.selection == "proportional":
                #eval_X = np.sort(eval_X, order = ['fitness'])
                #print(eval_X)
                prob_sel = eval_X['fitness']/eval_X['fitness'].sum()
                c_prob_sel = np.cumsum(prob_sel)
                probs = np.random.rand(1,X.shape[0])
                # print(probs)
                # print(prob_sel)
                # print(c_prob_sel)
                new_X = np.zeros(X.shape, dtype = np.int64)
                for j, prob in enumerate(probs[0, :]):
                    i = np.searchsorted(c_prob_sel, prob)
                    #print(i)
                    #print(X[i, :])
                    new_X[j, :] = X[i, :]
                #return new_X, solution
            elif self.selection == "tournament":
                new_X = np.zeros(X.shape, dtype = np.int64)
                #print(eval_X)
                for i in range(2):
                    perm = np.random.permutation(X.shape[0])
                    #print(perm)
                    for j in range(0, X.shape[0]//2):
                        index = [perm[2 * j], perm[2 * j + 1]]
                        max_ind = np.argmax([eval_X['fitness'][index[0]], eval_X['fitness'][index[1]]]) 
                        #print(max_ind)
                        new_X[i * X.shape[0]//2 + j, :] = X[index[max_ind], :]
                #return new_X, solution
            elif self.selection == "sus":
                prob = eval_X['fitness'].sum()/X.shape[0]
                #start = 
                pointers = [random.random()*prob + i*prob for i in range(X.shape[0])]
                new_X = np.zeros(X.shape, dtype = np.int64)
                cum_fitness = np.cumsum(eval_X['fitness'])
                for j, p in enumerate(pointers):
                    i = np.searchsorted(cum_fitness, p)
                    #print(i)
                    new_X[j, :] = X[i, :]
                #return new_X, solution


        else:
            elitism_num = math.floor(self.elitism * X.shape[0])
            if self.selection == "proportional":
                eval_X = np.sort(eval_X, order = ['fitness'])
                # print(eval_X)
                
                #prob_sel = eval_X['fitness'][0:elitism_num]/eval_X['fitness'][0:elitism_num].sum()
                prob_sel = eval_X['fitness']/eval_X['fitness'].sum()
                c_prob_sel = np.cumsum(prob_sel)
                probs = np.random.rand(1,X.shape[0] - elitism_num)
                # print(probs)
                # print(prob_sel)
                # print(c_prob_sel)
                new_X = np.zeros(X.shape, dtype = np.int64)
                new_X[0 : elitism_num, :] = X[eval_X['index'][eval_X.size - elitism_num: eval_X.size], :]
                #eval_X = eval_X[eval_X.size - elitism_num: eval_X.size]
                # print(eval_X)
                for j, prob in enumerate(probs[0, :]):
                    i = np.searchsorted(c_prob_sel, prob)
                    # print(i)
                    # print(X[eval_X['index'][i], :])
                    new_X[j + elitism_num, :] = X[eval_X['index'][i], :]
                #return new_X, solution
            elif self.selection == "tournament":
                new_X = np.zeros(X.shape, dtype = np.int64)
                new_X[0 : elitism_num, :] = X[eval_X['index'][eval_X.size - elitism_num: eval_X.size], :]
                #print(eval_X)
                for i in range(2):
                    perm = np.random.permutation(X.shape[0])
                    #print(perm)
                    for j in range(0, (X.shape[0] - elitism_num)//2):
                        index = [perm[2 * j], perm[2 * j + 1]]
                        max_ind = np.argmax([eval_X['fitness'][index[0]], eval_X['fitness'][index[1]]]) 
                        #print(max_ind)
                        new_X[i * (X.shape[0] - elitism_num)//2 + j + elitism_num, :] = X[index[max_ind], :]
                #return new_X, solution
            elif self.selection == "sus":
                prob = eval_X['fitness'].sum()/(X.shape[0] - elitism_num)
                #start = 
                pointers = [random.random()*prob + i*prob for i in range(X.shape[0] - elitism_num)]
                new_X = np.zeros(X.shape, dtype = np.int64)
                new_X[0 : elitism_num, :] = X[eval_X['index'][eval_X.size - elitism_num: eval_X.size], :]
                cum_fitness = np.cumsum(eval_X['fitness'])
                for j, p in enumerate(pointers):
                    i = np.searchsorted(cum_fitness, p)
                    #print(i)
                    new_X[j + elitism_num, :] = X[i, :]
        return new_X, solution





    
