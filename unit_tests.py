from test_genetic_algorithm import TestGeneticAlgorithm
from hklearn_genetic.problem import Rastrigin, Beale, Himmelblau, Eggholder
import numpy as np

### Proportional selection, no elitism
ga = TestGeneticAlgorithm([[0.25410149, 0.71410111, 0.31915886, 0.45725239]], pc = 0.9, pm = 0.5, max_iter=10)
rast = Rastrigin(n_dim=2, n_prec=0)
beale = Beale(n_prec=0)
himme = Himmelblau(n_prec=0)
egg = Eggholder(n_prec=0)
rast_init_pop = np.array([[1, 0, 0, 0, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 1, 0],
       [1, 1, 0, 1, 0, 1, 0, 1],
       [1, 0, 1, 0, 0, 0, 1, 1]])

beale_init_pop = np.array([[1, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 1, 1, 0],
       [0, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 0, 0, 1, 1, 0]])

himme_init_pop = np.array([[1, 1, 0, 1, 1, 0, 1, 0],
       [0, 1, 1, 0, 1, 1, 1, 1],
       [1, 0, 1, 1, 1, 0, 1, 1],
       [1, 0, 0, 0, 1, 0, 0, 1]])

egg_init_pop = np.array([[1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
       [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
       [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]])

# print(rast_init_pop)
# rast_selected = ga.select(rast, rast_init_pop)
# print(rast_selected)
# rast_cross = ga.crossover(rast_selected[0])
# print(rast_cross)
# print(ga.mutate(rast_cross))

# print(beale_init_pop)
# beale_selected = ga.select(beale, beale_init_pop)
# print(beale_selected)
# beale_crossover = ga.crossover(beale_selected[0])
# print(beale_crossover)
# print(ga.mutate(beale_crossover))

# print(himme_init_pop)
# himme_selected = ga.select(himme, himme_init_pop)
# print(himme_selected)
# himme_crossover = ga.crossover(himme_selected[0])
# print(himme_crossover)
# print(ga.mutate(himme_crossover))

# print(egg_init_pop)
# egg_selected = ga.select(egg, egg_init_pop)
# print(egg_selected)
# egg_crossover = ga.crossover(egg_selected[0])
# print(egg_crossover)
# print(ga.mutate(egg_crossover))


### Proportional selection, elitism
# ga = TestGeneticAlgorithm(pc = 0.9, pm = 0.5, max_iter=10, elitism = 0.4)
# print(rast_init_pop)
# rast_selected = ga.select(rast, rast_init_pop)
# print(rast_selected)
# rast_cross = ga.crossover(rast_selected[0])
# print(rast_cross)
# print(ga.mutate(rast_cross))

# print(beale_init_pop)
# beale_selected = ga.select(beale, beale_init_pop)
# print(beale_selected)
# beale_crossover = ga.crossover(beale_selected[0])
# print(beale_crossover)
# print(ga.mutate(beale_crossover))

# print(himme_init_pop)
# himme_selected = ga.select(himme, himme_init_pop)
# print(himme_selected)
# himme_crossover = ga.crossover(himme_selected[0])
# print(himme_crossover)
# print(ga.mutate(himme_crossover))

# print(egg_init_pop)
# egg_selected = ga.select(egg, egg_init_pop)
# print(egg_selected)
# egg_crossover = ga.crossover(egg_selected[0])
# print(egg_crossover)
# print(ga.mutate(egg_crossover))

### Tournament selection
ga = TestGeneticAlgorithm(pc = 0.9, pm = 0.5, max_iter=10, selection = "tournament")
print(rast_init_pop)
rast_selected = ga.select(rast, rast_init_pop)
print(rast_selected)
rast_cross = ga.crossover(rast_selected[0])
print(rast_cross)
print(ga.mutate(rast_cross))

print(beale_init_pop)
beale_selected = ga.select(beale, beale_init_pop)
print(beale_selected)
beale_crossover = ga.crossover(beale_selected[0])
print(beale_crossover)
print(ga.mutate(beale_crossover))

print(himme_init_pop)
himme_selected = ga.select(himme, himme_init_pop)
print(himme_selected)
himme_crossover = ga.crossover(himme_selected[0])
print(himme_crossover)
print(ga.mutate(himme_crossover))

print(egg_init_pop)
egg_selected = ga.select(egg, egg_init_pop)
print(egg_selected)
egg_crossover = ga.crossover(egg_selected[0])
print(egg_crossover)
print(ga.mutate(egg_crossover))

### Tournament selection w elitism
ga = TestGeneticAlgorithm(pc = 0.9, pm = 0.5, max_iter=10, elitism = 0.4, selection = "tournament")
print(rast_init_pop)
rast_selected = ga.select(rast, rast_init_pop)
print(rast_selected)
rast_cross = ga.crossover(rast_selected[0])
print(rast_cross)
print(ga.mutate(rast_cross))

print(beale_init_pop)
beale_selected = ga.select(beale, beale_init_pop)
print(beale_selected)
beale_crossover = ga.crossover(beale_selected[0])
print(beale_crossover)
print(ga.mutate(beale_crossover))

print(himme_init_pop)
himme_selected = ga.select(himme, himme_init_pop)
print(himme_selected)
himme_crossover = ga.crossover(himme_selected[0])
print(himme_crossover)
print(ga.mutate(himme_crossover))

print(egg_init_pop)
egg_selected = ga.select(egg, egg_init_pop)
print(egg_selected)
egg_crossover = ga.crossover(egg_selected[0])
print(egg_crossover)
print(ga.mutate(egg_crossover))

### SUS selection
ga = TestGeneticAlgorithm(pc = 0.9, pm = 0.5, max_iter=10, selection = "sus")
print(rast_init_pop)
rast_selected = ga.select(rast, rast_init_pop)
print(rast_selected)
rast_cross = ga.crossover(rast_selected[0])
print(rast_cross)
print(ga.mutate(rast_cross))

print(beale_init_pop)
beale_selected = ga.select(beale, beale_init_pop)
print(beale_selected)
beale_crossover = ga.crossover(beale_selected[0])
print(beale_crossover)
print(ga.mutate(beale_crossover))

print(himme_init_pop)
himme_selected = ga.select(himme, himme_init_pop)
print(himme_selected)
himme_crossover = ga.crossover(himme_selected[0])
print(himme_crossover)
print(ga.mutate(himme_crossover))

print(egg_init_pop)
egg_selected = ga.select(egg, egg_init_pop)
print(egg_selected)
egg_crossover = ga.crossover(egg_selected[0])
print(egg_crossover)
print(ga.mutate(egg_crossover))

### SUS selection w elitism
ga = TestGeneticAlgorithm(pc = 0.9, pm = 0.5, max_iter=10, elitism = 0.4)
print(rast_init_pop)
rast_selected = ga.select(rast, rast_init_pop)
print(rast_selected)
rast_cross = ga.crossover(rast_selected[0])
print(rast_cross)
print(ga.mutate(rast_cross))

print(beale_init_pop)
beale_selected = ga.select(beale, beale_init_pop)
print(beale_selected)
beale_crossover = ga.crossover(beale_selected[0])
print(beale_crossover)
print(ga.mutate(beale_crossover))

print(himme_init_pop)
himme_selected = ga.select(himme, himme_init_pop)
print(himme_selected)
himme_crossover = ga.crossover(himme_selected[0])
print(himme_crossover)
print(ga.mutate(himme_crossover))

print(egg_init_pop)
egg_selected = ga.select(egg, egg_init_pop)
print(egg_selected)
egg_crossover = ga.crossover(egg_selected[0])
print(egg_crossover)
print(ga.mutate(egg_crossover))