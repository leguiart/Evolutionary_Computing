import numpy as np
import random


def differential_evolution(problem, n_individuals, pc, F,  max_iter):
    X = problem.populate(n_individuals)
    its = 1
    avg_fitness_li = []
    best_fitness_li = []
    while its <= max_iter:
        X_eval = evaluate_population(problem, X)
        best_fitness_li += [np.sort(X_eval, axis=0)[-1][0]]
        avg_fitness_li += [np.average(X_eval, axis=0)]
        sol = problem.stop_criteria(X_eval)
        if len(sol) > 0:
            return X[sol, :], its, avg_fitness_li, best_fitness_li
        recomb = np.random.uniform(size = X.shape) <= pc
        for i in range(X.shape[0]):
            idxs = list(range(0, X.shape[0]))
            idxs.pop(i)
            x_is = random.sample(idxs, 3)
            v = X[x_is[0], :] + F*(X[x_is[1], :] - X[x_is[2], :])
            v = np.clip(v, problem.bounds[0], problem.bounds[1])
            u = X[i, :].copy()
            u[recomb[i,:]] = v[recomb[i,:]]
            if problem.evaluate(X[i, :]) >  problem.evaluate(u):
                X[i, :] = u.copy()
        its+=1
    X_eval = problem.evaluate(X)
    sol = problem.stop_criteria(X_eval)
    return [], its-1, avg_fitness_li, best_fitness_li


def evaluate_population(problem, X):
    pop = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        pop[i] =  problem.evaluate(X[i,:])
    return pop


