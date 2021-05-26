import numpy as np
from hklearn_genetic.problem import BaseProblem, _BaseRealProblem

class Particle:
    def __init__(self, n_dim):
        self.pos = np.zeros(n_dim)
        self.best_known = np.zeros(n_dim)
        self.vel = np.zeros(n_dim)

class Swarm:
    def __init__(self, n_individuals : int, bounds : tuple, n_dim : int):
        self.n_individuals = n_individuals
        self.bounds = bounds
        self.n_dim = n_dim
        self.best_known = np.random.uniform(self.bounds[0], self.bounds[1] + 0.1, size = self.n_dim)
        self.particles = np.array([Particle(self.n_dim) for i in range(n_individuals)])
    
    def initialize_population(self, problem):
        for p_i in self.particles:
            p_i.pos = np.random.uniform(self.bounds[0], self.bounds[1] + 0.1, size = self.n_dim)
            p_i.best_known = p_i.pos.copy()
            if problem.evaluate(p_i.best_known) < problem.evaluate(self.best_known):
                self.best_known = p_i.best_known
            rang = abs(self.bounds[1] - self.bounds[0])
            p_i.vel = np.random.uniform(-rang, rang, size = self.n_dim)

    def get_solutions(self, problem):
        n =  self.particles[0].best_known.size
        pop = np.zeros((len(self.particles), 1))
        
        for i in range(len(self.particles)):
            pop[i] =  problem.evaluate(self.particles[i].best_known)
        #pop[len(self.particles)] = problem.evaluate(self.best_known)
        return problem.stop_criteria(pop), problem.evaluate(self.best_known) <= problem.thresh

class BaseRastriginPSO(BaseProblem):
    def a_eval(self, X_decoded):
        return 10.*self.n_dim + np.sum(X_decoded**2 - 10.*np.cos(2.*np.pi*X_decoded))

class BaseBealePSO(BaseProblem):
    def a_eval(self, X_decoded):
        first_term = (1.5 - X_decoded[0] + X_decoded[0]*X_decoded[1])**2
        second_term = (2.25 - X_decoded[0] + X_decoded[0]*(X_decoded[1]**2))**2
        third_term = (2.625 - X_decoded[0] + X_decoded[0]*(X_decoded[1]**3))**2
        return first_term + second_term + third_term

class BaseHimmelblauPSO(BaseProblem):
    def a_eval(self, X_decoded):
        first_term = (X_decoded[0]**2 + X_decoded[1] - 11.)**2
        second_term = (X_decoded[0] + X_decoded[1]**2 - 7.)**2
        return first_term + second_term

class BaseEggholderPSO(BaseProblem):
    def a_eval(self, X_decoded):
        first_term = - (X_decoded[1] + 47.)*np.sin(np.sqrt(np.abs(X_decoded[0]/2. + (X_decoded[1] + 47.))))
        second_term =  - X_decoded[0]*np.sin(np.sqrt(np.abs(X_decoded[0] - (X_decoded[1] + 47.))))
        return first_term + second_term

class RealRastriginPSO(_BaseRealProblem, BaseRastriginPSO):
    def __init__(self, n_dim = 2):
        super().__init__(0.01, (-5.12, 5.12), n_dim=n_dim)
    def stop_criteria(self, X_eval):
        return list(np.where(X_eval <= self.thresh)[0])

class RealBealePSO(_BaseRealProblem, BaseBealePSO):
    def __init__(self):
        super().__init__(0.01, (-4.5, 4.5), n_dim=2)
    def stop_criteria(self, X_eval):
        return list(np.where(X_eval <= self.thresh)[0])

class RealHimmelblauPSO(_BaseRealProblem, BaseHimmelblauPSO):
    def __init__(self):
        super().__init__(0.01, (-5., 5.), n_dim=2)
    def stop_criteria(self, X_eval):
        return list(np.where(X_eval <= self.thresh)[0])

class RealEggholderPSO(_BaseRealProblem, BaseEggholderPSO):
    def __init__(self):
        super().__init__(-957., (-512., 512.), n_dim=2)
    def stop_criteria(self, X_eval):
        return list(np.where(X_eval <= self.thresh)[0])

def pso(num_individuals, problem, max_iter, omega, phi_p, phi_g, lr, b, fr):
    S = Swarm(num_individuals, problem.bounds, problem.n_dim)
    S.initialize_population(problem)
    iter = 0
    solutions = []
    swarm_best_sol = False
    best_knowns = []
    p_best_knowns = []
    while iter <  max_iter and len(solutions) == 0 and not swarm_best_sol:
        best_knowns += [S.best_known]
        p_best_knowns += [np.average([problem.evaluate(p_i.best_known) for p_i in S.particles])]
        for p_i in S.particles:           
            r_pg = np.random.uniform(size=(2, problem.n_dim))
            p_i.vel = omega*p_i.vel + phi_p*r_pg[0, :]*(p_i.best_known - p_i.pos) + phi_g*r_pg[1, :]*(S.best_known - p_i.pos)
            p_i.vel = p_i.vel*(1 - fr)
            p_i.pos += lr*p_i.vel
            for i in range(len(p_i.pos)):
                if p_i.pos[i] > problem.bounds[1]:
                    p_i.vel[i] = -b * p_i.vel[i]
                    p_i.pos[i] = problem.bounds[1]
                    p_i.pos[i] += lr*p_i.vel[i]
                elif p_i.pos[i] < problem.bounds[0]:
                    p_i.vel[i] = -b * p_i.vel[i]
                    p_i.pos[i] = problem.bounds[0]
                    p_i.pos[i] += lr*p_i.vel[i]
                    
            if problem.evaluate(p_i.pos) < problem.evaluate(p_i.best_known):
                p_i.best_known = p_i.pos.copy()
                if problem.evaluate(p_i.pos) < problem.evaluate(S.best_known):
                    S.best_known = p_i.best_known.copy()
        solutions, swarm_best_sol = S.get_solutions(problem)
        iter+=1
    for i in range(len(best_knowns)):
        best_knowns[i] = problem.evaluate(best_knowns[i])
    if swarm_best_sol:
        p_best_sols = [p.best_known for p in S.particles[solutions]] + [S.best_known]
    else:
        p_best_sols = [p.best_known for p in S.particles[solutions]]
    #print([p.pos for p in S.particles])
    return  p_best_sols, iter, S.best_known, best_knowns, p_best_knowns


