import abc
import numpy as np
import math
import random
import itertools as it
from hklearn_genetic.board_conflicts import conflict
from deap import tools, gp

class ProblemInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'evaluate') and 
                callable(subclass.evaluate) and
                hasattr(subclass, 'stop_criteria') and
                callable(subclass.stop_criteria) and
                hasattr(subclass, 'populate') and
                callable(subclass.populate) and 
                hasattr(subclass, 'decode') and
                callable(subclass.decode) and 
                hasattr(subclass, 'crossover') and 
                callable(subclass.crossover) and
                hasattr(subclass, 'mutate') and 
                callable(subclass.mutate))


@ProblemInterface.register
class IProblem:
    """Evalua las soluciones potenciales del problema"""
    def evaluate(self, X):
        pass

    """Regresa si la población ha llegado al criterio de paro"""
    def stop_criteria(self, X_eval):
        pass

    """Crea una poblacion inicial de posibles soluciones"""
    def populate(self, n_individuals):
        pass

    """Pasa a la población del genotipo al fenotipo"""
    def decode(self, X_encoded):
        pass

    """Efectúa la cruza con los elementos de la población"""
    def crossover(self, X, pc, elitism):
        pass

    """Efectúa la mutación con los elementos de la población"""
    def mutate(self, X, pm, elitism):
        pass


class BaseProblem(IProblem):
    def get_crossover_probs(self, n_cross):
        return np.random.rand(1 , n_cross)[0,:]

    def get_crossover_points(self, length):
        return np.random.randint(0, length)

    @abc.abstractmethod
    def a_eval(self, X_decoded):
        pass

    def evaluate(self, X):
        decoded_rep = self.decode(X)
        X_eval = self.a_eval(decoded_rep)
        return X_eval

    def crossover(self, X, pc, elitism):
        if not elitism:
            n_cross = X.shape[0] // 2
            elitism_num = 0
        else:
            elitism_num = math.floor(elitism * X.shape[0])
            n_cross = (X.shape[0] - elitism_num) // 2
        prob_cross = self.get_crossover_probs(n_cross)
        for i, p in enumerate(prob_cross):
            if p <= pc:
                cross_point = self.get_crossover_points(X.shape[1] - 1)
                son1 = X[2*i + elitism_num,:].copy()
                son2 = X[2*i + 1 + elitism_num, :].copy()
                son1[cross_point : X.shape[1]] = X[2*i + 1 + elitism_num, cross_point : X.shape[1]].copy()
                son2[cross_point : X.shape[1]] = X[2*i + elitism_num, cross_point : X.shape[1]].copy()
                X[2*i + elitism_num,:] = son1
                X[2*i + 1 + elitism_num,:] = son2
        return X


class _BaseGeneticProgrammingProblem(BaseProblem):
    def __init__(self, mutation_type = "Branch"):
        self.avg_lengths = []
        self.mutation_type = mutation_type

    def populate(self, n_individuals):
        return tools.initRepeat(list, lambda: gp.genHalfAndHalf(self.pset, min_=1, max_=2), n_individuals)

    def decode(self, X_encoded):
        X_decoded = []
        length_sum = 0
        for x_i in X_encoded:
            tree = gp.PrimitiveTree(x_i)
            length_sum += len(tree)
            X_decoded += [gp.compile(tree, self.pset)]
        self.avg_lengths += [length_sum/len(X_decoded)]
        return X_decoded

    def crossover(self, X, pc, elitism):
        if not elitism:
            n_cross = len(X) // 2
            elitism_num = 0
        else:
            elitism_num = math.floor(elitism * len(X))
            n_cross = (len(X) - elitism_num) // 2
        prob_cross = self.get_crossover_probs(n_cross)
        for i, p in enumerate(prob_cross):
            if p <= pc:
                parent1 = gp.PrimitiveTree(X[2*i + elitism_num])
                parent2 = gp.PrimitiveTree(X[2*i + 1 + elitism_num])
                offspring = gp.cxOnePoint(parent1, parent2)
                if offspring[0].height < self.height_limit:
                    X[2*i + elitism_num] = offspring[0]
                else:
                    r = random.uniform(0, 1)
                    X[2*i + elitism_num] = X[2*i + 1 + elitism_num].copy() if r >= 0.5 else X[2*i + elitism_num]

                if offspring[1].height < self.height_limit:
                    X[2*i + 1 + elitism_num] = offspring[1]
                else:
                    r = random.uniform(0, 1)
                    X[2*i + 1 + elitism_num] = X[2*i + elitism_num].copy() if r >= 0.5 else X[2*i + 1 + elitism_num]
        return X

    def mutate(self, X, pm, elitism):
        if pm > 0:
            mutate_m = np.random.uniform(size = (len(X), 1))
            mutate_m = mutate_m <= pm
            func = lambda pset, type_ : gp.genFull(pset, min_=0, max_=2)
            if not elitism:
                for i, m  in enumerate(mutate_m):
                    #if m <= 1./len(X[i]):
                    if m:
                        if self.mutation_type == "Branch":
                            offspring = gp.mutUniform(gp.PrimitiveTree(X[i]), func, self.pset)
                        elif self.mutation_type == "Node":
                            offspring = gp.mutNodeReplacement(gp.PrimitiveTree(X[i]), self.pset)       
                        if offspring[0].height <= self.height_limit:
                            X[i] = offspring[0]
            else:
                elitism_num = math.floor(elitism * len(X))
                for i in range(elitism_num, len(X)):
                    #if mutate_m[i] <= 1./len(X[i]):
                    if mutate_m[i]:
                        if self.mutation_type == "Branch":
                            offspring = gp.mutUniform(gp.PrimitiveTree(X[i]), func, self.pset)
                        elif self.mutation_type == "Node":
                            offspring = gp.mutNodeReplacement(gp.PrimitiveTree(X[i]), self.pset)  
                        if offspring[0].height <= self.height_limit:
                            X[i] = offspring[0]
        return X


class SymbolicRegressionProblem(_BaseGeneticProgrammingProblem):
    def __init__(self, bounds, pset, real_values, height_limit, mutation_type = "Branch"):
        super().__init__(mutation_type)
        self.bounds = bounds
        self.height_limit = height_limit
        self.pset = pset
        self.real_values = real_values
        self.points = np.linspace(self.bounds[0], self.bounds[1], num = len(real_values))

    def a_eval(self, X_decoded, X_encoded):
        m = len(self.points)
        X_fitness = []
        for j, func in enumerate(X_decoded):
            try:
                s = 0
                for i in range(m):
                    s += (func(self.points[i]) - self.real_values[i])**2
                X_fitness += [- (1./m)*s]
            except Exception as e:
                print(e)
                x_encoded = X_encoded[j]
                print(gp.PrimitiveTree(x_encoded))
        return X_fitness

    def stop_criteria(self, X_eval):
        return list(np.where(X_eval >= - 0.2)[0])


class BitParityCheck(_BaseGeneticProgrammingProblem):
    def __init__(self, pset, real_values, height_limit, mutation_type = "Branch"):
        super().__init__(mutation_type)
        self.height_limit = height_limit
        self.pset = pset
        self.real_values = real_values
        self.points = list(map(list, it.product([False, True], repeat=int(math.log2(len(self.real_values))))))

    def stop_criteria(self, X_eval):
        return list(np.where(X_eval >= 0)[0])  

    def a_eval(self, X_decoded, X_encoded):
        m = len(self.points)
        X_fitness = []
        for j, func in enumerate(X_decoded):
            try:
                X_fitness += [-sum(func(*in_) == out for in_, out in zip(self.points, self.real_values))]
            except Exception as e:
                print(e)
                print(gp.PrimitiveTree(X_encoded[j]))

        return X_fitness


class NeutralityProblem(_BaseGeneticProgrammingProblem):
    def __init__(self, pset, T, height_limit, terminals, mutation_type = "Branch"):
        super().__init__(mutation_type)
        self.height_limit = height_limit
        self.pset = pset
        self.T = T
        self.str_terminals = [str(t) for t in terminals]
        for t in terminals:
            self.pset.addTerminal(t)
        self.gene_counts = {t : [] for t in self.str_terminals}
        
    def stop_criteria(self, X_eval):
        return []

    def a_eval(self, X_decoded, X_encoded):
        X_fitness = []
        for j, x_i in enumerate(X_decoded):
            try:
                X_fitness += [-abs(self.T - x_i)]
            except Exception as e:
                print(e)
                print(gp.PrimitiveTree(X_encoded[j]))

        for gene in self.gene_counts.keys():
            self.gene_counts[gene]+=[0]
        for x in X_encoded:
            x_tree = gp.PrimitiveTree(x)
            x_tree_str = str(x_tree)
            for s in x_tree_str:
                if s in self.str_terminals:
                    self.gene_counts[s][-1] += 1

        return X_fitness
       

class _BaseBinaryProblem(BaseProblem):
    def __init__(self, thresh, bounds, n_dim = 2, n_prec = 4):
        self.bounds = bounds
        self.n_dim = n_dim
        self.gene_length = math.ceil(math.log2((self.bounds[1] - self.bounds[0])*10**n_prec))
        self.thresh = thresh

    def stop_criteria(self, X_eval):
        return list(np.where(X_eval >= self.thresh)[0])

    def populate(self, n_individuals):
        return np.random.randint(2, size = (n_individuals, self.gene_length*self.n_dim))

    def decode(self, X_encoded):
        decoded_rep = np.zeros((X_encoded.shape[0], self.n_dim))
        for i in range(self.n_dim):
            decoded_rep[:,i] = (X_encoded[:, i*self.gene_length : (i + 1)*self.gene_length]@(2**np.arange(X_encoded[:, i*self.gene_length : (i + 1)*self.gene_length].shape[1], dtype = np.float64)[::-1][:, np.newaxis])).T
        return self.bounds[0] + decoded_rep*(self.bounds[1] - self.bounds[0])/(2**self.gene_length - 1)

    def get_mutation(self, shape):
        return np.random.uniform(size = shape)
    
    def mutate(self, X, pm, elitism):
        mutate_m = self.get_mutation((X.shape[0], X.shape[1]))
        mutate_m = mutate_m <= pm
        X_bit = X == 1
        if not elitism:
            X = np.logical_xor(X_bit, mutate_m)
        else:
            elitism_num = math.floor(elitism * X.shape[0])           
            X[elitism_num :  X.shape[0], :] = np.logical_xor(X_bit, mutate_m)[elitism_num :  X.shape[0], :]
        X = X.astype(int)
        return X


class _BaseIntegerProblem(BaseProblem):
    def __init__(self, thresh, n_dim = 2):
        self.n_dim = n_dim
        self.thresh = thresh

    def stop_criteria(self, X_eval):
        return list(np.where(X_eval >= self.thresh)[0])

    def populate(self, n_individuals):
        return np.random.randint(self.n_dim, size = (n_individuals, self.n_dim))

    def decode(self, X_encoded):
        return X_encoded

    def get_mutation(self, shape):
        return np.random.uniform(size = shape)
    
    def mutate(self, X, pm, elitism):
        mutate_m = self.get_mutation((X.shape[0], 1))
        mutate_m = mutate_m <= pm
        if not elitism:
            for i, m  in enumerate(mutate_m):
                if m:
                    indices = np.random.permutation(X.shape[1])[0 : 2]
                    X[i,indices[0]], X[i, indices[1]] = X[i, indices[1]], X[i, indices[0]]
        else:
            elitism_num = math.floor(elitism * X.shape[0])
            for i in range(elitism_num, X.shape[0]):
                if mutate_m[i]:
                    indices = np.random.permutation(X.shape[1])[0 : 2]
                    X[i,indices[0]], X[i, indices[1]] = X[i, indices[1]], X[i, indices[0]]
        return X


class _BaseRealProblem(BaseProblem):
    def __init__(self, thresh, bounds, n_dim = 2):
        self.n_dim = n_dim
        self.thresh = thresh
        self.bounds = bounds

    def stop_criteria(self, X_eval):
        return list(np.where(X_eval >= self.thresh)[0])

    def populate(self, n_individuals):
        return np.random.uniform(self.bounds[0], self.bounds[1] + 0.1, size = (n_individuals, self.n_dim))

    def decode(self, X_encoded):
        return X_encoded

    def get_crossover_points(self, length):
        return np.random.uniform(low = -.25 , high = 1.25, size = length)

    def crossover(self, X, pc, elitism):
        if not elitism:
            n_cross = X.shape[0] // 2
            elitism_num = 0
        else:
            elitism_num = math.floor(elitism * X.shape[0])
            n_cross = (X.shape[0] - elitism_num) // 2
        prob_cross = self.get_crossover_probs(n_cross)
        for i, p in enumerate(prob_cross):
            if p <= pc:
                alphas = self.get_crossover_points(X.shape[1])
                X[2*i + elitism_num,:] += alphas * (X[2*i + 1 + elitism_num, :] - X[2*i + elitism_num,:])
                X[2*i + 1 + elitism_num,:] += alphas * (X[2*i + elitism_num,:] - X[2*i + 1 + elitism_num, :])
        return X

    def get_mutation(self, shape):
        return np.random.uniform(size = shape)
    
    def mutate(self, X, pm, elitism):
        if not elitism:
            elitism = 0

        rang = (self.bounds[1] - self.bounds[0])*.0001
        mutate_m = self.get_mutation((X.shape[0], X.shape[1]))
        
        mutate_plus_minus = self.get_mutation((X.shape[0], X.shape[1]))

        mutate_m[mutate_m <= pm] = 1.
        mutate_m[mutate_m < 1.] = 0.
        mutate_plus_minus[mutate_plus_minus <= .5] = 1.0
        mutate_plus_minus[mutate_plus_minus > .5] = -1.0
        
        elitism_num = math.floor(elitism * X.shape[0])
        for i in range(elitism_num, X.shape[0]):
            mutate_delta = self.get_mutation((X.shape[1], X.shape[1]))
            mutate_delta[mutate_delta <= 1./self.n_dim] = 1.
            mutate_delta[mutate_delta < 1.] = 0.
            deltas = (mutate_delta @ (2**-np.arange(self.n_dim, dtype = np.float64)[:, np.newaxis])).T
            X[i, :] = X[i, :] + mutate_m[i, :] * mutate_plus_minus[i, :] * rang * deltas

        return X

class BaseNQueen(BaseProblem):
    def a_eval(self, X_decoded):
        X_fitness = np.zeros(X_decoded.shape[0])
        for i, x in enumerate(X_decoded):
            X_fitness[i] = -conflict(x)
        #print(X_fitness)
        return np.array(list(zip(X_fitness, list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])


class IntegerNQueen(_BaseIntegerProblem, BaseNQueen):
    def __init__(self, n_dim = 2):
        super().__init__(0, n_dim = n_dim)


class RealNQueen(_BaseRealProblem, BaseNQueen):
    def __init__(self, n_dim = 2):
        super().__init__(0, (0, 5.), n_dim = n_dim)

    def decode(self, X_encoded):
        X_decoded = np.zeros(X_encoded.shape, dtype=np.int64)
        for i, x in enumerate(X_encoded):
            indexed = np.array(list(zip(x, list(range(X_decoded.shape[1])))), dtype = [('real_rep', float),('index', int)])
            indexed = np.sort(indexed, order=["real_rep"])
            X_decoded[i, :] = indexed["index"]
        return X_decoded


class BinaryNQueen(_BaseBinaryProblem, BaseNQueen):
    def __init__(self, n_dim = 2, n_prec = 4):
        super().__init__(0, (0.01, n_dim), n_dim = n_dim, n_prec=n_prec)
    
    def decode(self, X_encoded):
        return np.ceil(super().decode(X_encoded)).astype(int) - 1


class BaseRastrigin(BaseProblem):
    def __init__(self):
        self.rank = 100.

    def a_eval(self, X_decoded):
        return np.array(list(zip(self.rank - (10.*self.n_dim + np.sum(X_decoded**2 - 10.*np.cos(2.*np.pi*X_decoded), axis = 1)), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])

class BaseBeale(BaseProblem):
    def __init__(self):
        self.rank = 150000.
    def a_eval(self, X_decoded):
        first_term = (1.5 - X_decoded[:, 0] + X_decoded[:, 0]*X_decoded[:, 1])**2
        second_term = (2.25 - X_decoded[:, 0] + X_decoded[:, 0]*(X_decoded[:, 1]**2))**2
        third_term = (2.625 - X_decoded[:, 0] + X_decoded[:, 0]*(X_decoded[:, 1]**3))**2
        return np.array(list(zip(self.rank - (first_term + second_term + third_term), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])


class BaseHimmelblau(BaseProblem):
    def __init__(self):
        self.rank = 2200.
        
    def a_eval(self, X_decoded):
        first_term = (X_decoded[:, 0]**2 + X_decoded[:, 1] - 11.)**2
        second_term = (X_decoded[:, 0] + X_decoded[:, 1]**2 - 7.)**2
        return np.array(list(zip(self.rank - (first_term + second_term), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])


class BaseEggholder(BaseProblem):
    def __init__(self):
        self.rank = 1200.

    def a_eval(self, X_decoded):
        first_term = - (X_decoded[:, 1] + 47)*np.sin(np.sqrt(np.abs(X_decoded[:, 0]/2. + (X_decoded[:, 1] + 47))))
        second_term =  - X_decoded[:, 0]*np.sin(np.sqrt(np.abs(X_decoded[:, 0] - (X_decoded[:, 1] + 47))))
        return np.array(list(zip(self.rank - (first_term + second_term), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])


class BinaryRastrigin(_BaseBinaryProblem, BaseRastrigin):
    def __init__(self, n_dim = 2, n_prec = 4):
        super().__init__(99.99, (-5.12, 5.12), n_dim=n_dim, n_prec=n_prec)
        BaseRastrigin.__init__(self)

class BinaryBeale(_BaseBinaryProblem, BaseBeale):
    def __init__(self, n_prec = 4):
        super().__init__(149999.99, (-4.5, 4.5), n_dim=2, n_prec=n_prec)
        BaseBeale.__init__(self)

class BinaryHimmelblau(_BaseBinaryProblem, BaseHimmelblau):
    def __init__(self, n_prec = 4):
        super().__init__(2199.99, (-5., 5.), n_dim=2, n_prec=n_prec)
        BaseHimmelblau.__init__(self)
class BinaryEggholder(_BaseBinaryProblem, BaseEggholder):
    def __init__(self, n_prec = 4):
        super().__init__(2157., (-512., 512.), n_dim=2, n_prec=n_prec)
        BaseEggholder.__init__(self)


class RealRastrigin(_BaseRealProblem, BaseRastrigin):
    def __init__(self, n_dim = 2):
        super().__init__(99.99, (-5.12, 5.12), n_dim=n_dim)
        BaseRastrigin.__init__(self)

class RealBeale(_BaseRealProblem, BaseBeale):
    def __init__(self):
        super().__init__(149999.99, (-4.5, 4.5), n_dim=2)
        BaseBeale.__init__(self)

class RealHimmelblau(_BaseRealProblem, BaseHimmelblau):
    def __init__(self):
        super().__init__(2199.99, (-5., 5.), n_dim=2)
        BaseHimmelblau.__init__(self)

class RealEggholder(_BaseRealProblem, BaseEggholder):
    def __init__(self):
        super().__init__(2157., (-512., 512.), n_dim=2)
        BaseEggholder.__init__(self)



class RealRastriginPSO(_BaseRealProblem):
    def __init__(self, n_dim = 2):
        super().__init__(99.99, (-5.12, 5.12), n_dim=n_dim)

class RealBealePSO(_BaseRealProblem):
    def __init__(self):
        super().__init__(149999.99, (-4.5, 4.5), n_dim=2)

class RealHimmelblauPSO(_BaseRealProblem):
    def __init__(self):
        super().__init__(2199.99, (-5., 5.), n_dim=2)

class RealEggholderPSO(_BaseRealProblem):
    def __init__(self):
        super().__init__(2157., (-512., 512.), n_dim=2)