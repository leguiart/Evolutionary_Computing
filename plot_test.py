from utils import plot_superposed
import matplotlib
import numpy as np


y_s = [([4, 5, 6], 'r'),([6, 5, 4], 'b')]
x = [1, 2, 3]

plot_superposed(x, y_s)

f = open('datos_examen_2021II.txt', 'r')
data = [[]]
for l in f:
    data[0].append(float(l))
x = list(range(len(data[0])))
plot_superposed(x, data)