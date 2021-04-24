from hklearn_genetic.genetic_algorithm import GeneticAlgorithm

class TestRouletteGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, r_probs, pc = 0.6, pm = 0.1, max_iter = 5000, elitism = None, selection = "proportional"):
        super().__init__(pc = pc, pm = pm, max_iter = max_iter, elitism = elitism, selection = selection)
        self.roulette_probs = r_probs
        # self.crossover_points = c_points
        # self.crossover_probs = c_probs
        # self.mutation_probs = m_probs

    def set_roulette_probs(self, r_probs):
        self.roulette_probs = r_probs
        #return np.array([[0.25410149, 0.71410111, 0.31915886, 0.45725239]])

    # def set_mutation(self, m_probs):
    #     self.mutation_probs = m_probs

    # def set_crossover_probs(self, c_probs):
    #     self.crossover_probs = c_probs

    # def set_crossover_points(self, c_points):
    #     self.crossover_points = c_points

    def get_roulette_probs(self, n_individuals):
        return self.roulette_probs
        #return np.array([[0.25410149, 0.71410111, 0.31915886, 0.45725239]])

    # def get_mutation(self, shape):
    #     return self.mutation_probs

    # def get_crossover_probs(self, n_cross):
    #     return self.crossover_probs

    # def get_crossover_points(self, length):
    #     return self.crossover_points

class TestSinglePointCrossover(GeneticAlgorithm):
    def __init__(self, c_points, c_probs, pc = 0.6, pm = 0.1, max_iter = 5000, elitism = None, selection = "proportional"):
        super().__init__(pc = pc, pm = pm, max_iter = max_iter, elitism = elitism, selection = selection)
        # self.roulette_probs = r_probs
        self.crossover_points = c_points
        self.crossover_probs = c_probs
        # self.mutation_probs = m_probs


        #return np.array([[0.25410149, 0.71410111, 0.31915886, 0.45725239]])
    # def set_roulette_probs(self, r_probs):
    #     self.roulette_probs = r_probs
    # def set_mutation(self, m_probs):
    #     self.mutation_probs = m_probs

    def set_crossover_probs(self, c_probs):
        self.crossover_probs = c_probs

    def set_crossover_points(self, c_points):
        self.crossover_points = c_points

    # def get_roulette_probs(self, n_individuals):
    #     return self.roulette_probs
    #     #return np.array([[0.25410149, 0.71410111, 0.31915886, 0.45725239]])

    # def get_mutation(self, shape):
    #     return self.mutation_probs

    def get_crossover_probs(self, n_cross):
        return self.crossover_probs

    def get_crossover_points(self, length):
        return self.crossover_points