import abc
import numpy as np
import math

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
                callable(subclass.decode))


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

class _BaseFuncProblem(IProblem):
    def __init__(self, thresh, bounds, n_dim = 2, n_prec = 4):
        self.bounds = bounds
        self.n_dim = n_dim
        self.gene_length = math.ceil(math.log2((self.bounds[1] - self.bounds[0])*10**n_prec))
        self.thresh = thresh

    def evaluate(self, X):
        decoded_rep = self.decode(X)
        #X_eval = np.array(list(zip(1./(1. + 10.*self.n_dim + np.sum(decoded_rep**2 - 10.*np.cos(2.*np.pi*decoded_rep), axis = 1)), list(range(X.shape[0])))), dtype = [('fitness', float),('index', int)])
        X_eval = self.a_eval(decoded_rep)
        return X_eval

    @abc.abstractmethod
    def a_eval(self, X_decoded):
        pass

    def stop_criteria(self, X_eval):
        return list(np.where(X_eval >= self.thresh)[0])

    def populate(self, n_individuals):
        return np.random.randint(2, size = (n_individuals, self.gene_length*self.n_dim))

    def decode(self, X_encoded):
        decoded_rep = np.zeros((X_encoded.shape[0], self.n_dim))
        for i in range(self.n_dim):
            decoded_rep[:,i] = (X_encoded[:, i*self.gene_length : (i + 1)*self.gene_length]@(2**np.arange(X_encoded[:, i*self.gene_length : (i + 1)*self.gene_length].shape[1], dtype = np.float64)[::-1][:, np.newaxis])).T
        return self.bounds[0] + decoded_rep*(self.bounds[1] - self.bounds[0])/(2**self.gene_length - 1)

class Rastrigin(_BaseFuncProblem):
    def __init__(self, n_dim = 2, n_prec = 4):
        super().__init__(79.999, (-5.12, 5.12), n_dim=n_dim, n_prec=n_prec)


    def a_eval(self, X_decoded):
        return np.array(list(zip(80. - (10.*self.n_dim + np.sum(X_decoded**2 - 10.*np.cos(2.*np.pi*X_decoded), axis = 1)), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])

class Beale(_BaseFuncProblem):
    def __init__(self, n_prec = 4):
        super().__init__(149999.9999, (-4.5, 4.5), n_dim=2, n_prec=n_prec)

    def a_eval(self, X_decoded):
        first_term = (1.5 - X_decoded[:, 0] + X_decoded[:, 0]*X_decoded[:, 1])**2
        second_term = (2.25 - X_decoded[:, 0] + X_decoded[:, 0]*(X_decoded[:, 1]**2))**2
        third_term = (2.625 - X_decoded[:, 0] + X_decoded[:, 0]*(X_decoded[:, 1]**3))**2
        return np.array(list(zip(150000. - (first_term + second_term + third_term), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])

class Himmelblau(_BaseFuncProblem):
    def __init__(self, n_prec = 4):
        super().__init__(1999.999, (-5., 5.), n_dim=2, n_prec=n_prec)

    def a_eval(self, X_decoded):
        first_term = (X_decoded[:, 0]**2 + X_decoded[:, 1] - 11.)**2
        second_term = (X_decoded[:, 0] + X_decoded[:, 1]**2 - 7.)**2
        return np.array(list(zip(2000. - (first_term + second_term), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])

class Eggholder(_BaseFuncProblem):
    def __init__(self, n_prec = 4):
        super().__init__(1959., (-512., 512.), n_dim=2, n_prec=n_prec)


    def a_eval(self, X_decoded):
        first_term = - (X_decoded[:, 1] + 47)*np.sin(np.sqrt(np.abs(X_decoded[:, 0]/2. + (X_decoded[:, 1] + 47))))
        second_term =  - X_decoded[:, 0]*np.sin(np.sqrt(np.abs(X_decoded[:, 0] - (X_decoded[:, 1] + 47))))
        return np.array(list(zip(1000. - (first_term + second_term), list(range(X_decoded.shape[0])))), dtype = [('fitness', float),('index', int)])