from deap import tools, gp
from utils import average_list_of_lists, protectedDiv, protectedSqrt, protectedExp, sin, cos, tan, plot_superposed
import operator
import math
import random
import numpy as np
import itertools as it

def mse(func, points, y):
    s = 0
    for i in range(len(points)):
        s += (func(*points[i]) - y[i])
    return s/len(points)

y = []
f = open('datos_examen_2021II.txt', 'r')
data = [[]]
for l in f:
    y.append(float(l))

### Primitivas para regresión simbólica
#Creamos un conjunto de primitivas llamado "main"
pset = gp.PrimitiveSet("main", 1)
#Conjunto de parametros 2
#Agregamos no terminales, primer argumento es la función, segundo argumento es la aridad
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
#pset.addPrimitive(math.tan, 1)
#pset.addPrimitive(protectedExp, 1)
pset.addPrimitive(protectedSqrt, 1)
pset.addPrimitive(operator.abs, 1)

#Agregamos terminales (constantes)
# for i in range(1, 2):
#     pset.addTerminal(float(i))
#pset.addTerminal(math.pi)

#Agregamos terminales (argumentos), al agregar nuestras primitivas con su aridad, se crearon
#2 argumentos por defecto, los renombramos como 'x' y 'y'
pset.renameArguments(ARG0="x")
#pset.addEphemeralConstant('eff1', lambda: random.uniform(-2.5, 0))

solution = "mul(add(add(x, add(add(add(sin(x), add(mul(add(x, x), sin(x)), add(add(x, x), add(cos(add(x, x)), add(cos(add(x, x)), protectedSqrt(sin(x))))))), add(protectedSqrt(x), add(add(add(add(x, -0.4194563256837087), add(cos(cos(add(x, x))), add(cos(cos(add(x, add(cos(x), x)))), x))), add(add(x, add(add(add(x, x), add(x, x)), add(add(x, cos(-0.6014588440741915)), add(add(x, x), x)))), add(mul(add(x, x), sin(x)), -0.4194563256837087))), add(add(add(add(cos(add(add(abs(x), x), cos(add(x, x)))), add(x, add(protectedSqrt(x), add(cos(add(x, x)), x)))), add(add(add(x, add(sin(x), x)), abs(mul(-0.8635297113495377, -1.3107186753747198))), x)), add(x, add(add(add(x, add(x, x)), cos(add(x, x))), x))), add(sin(x), x))))), add(sin(x), add(add(cos(add(cos(add(x, x)), x)), add(add(cos(add(cos(-0.4194563256837087), add(x, x))), add(cos(add(x, add(cos(add(x, add(x, -0.4194563256837087))), x))), neg(-1.899702710288039))), add(cos(sin(x)), add(cos(add(x, x)), x)))), add(cos(add(add(add(x, x), add(add(x, x), add(x, x))), protectedSqrt(add(x, add(cos(add(add(x, add(x, -0.4194563256837087)), protectedSqrt(x))), add(x, add(x, -0.6014588440741915))))))), add(x, x)))))), add(add(add(x, x), cos(add(x, x))), add(add(add(x, add(add(cos(add(x, x)), add(sin(cos(add(add(add(x, x), add(x, add(x, -0.4194563256837087))), protectedSqrt(x)))), add(x, add(cos(add(add(add(x, x), add(x, add(x, -0.4194563256837087))), protectedSqrt(x))), x)))), add(add(-1.0998860715047434, x), x))), add(add(x, add(add(x, add(cos(add(x, x)), add(cos(add(add(add(x, x), neg(x)), protectedSqrt(x))), add(add(x, x), x)))), x)), add(add(add(x, add(add(x, -0.4194563256837087), x)), add(add(x, add(add(x, add(cos(cos(-0.6014588440741915)), x)), add(add(add(add(x, x), cos(add(x, x))), protectedSqrt(x)), x))), add(cos(add(cos(add(add(x, -0.4194563256837087), x)), add(x, x))), add(cos(add(add(add(x, x), add(x, x)), -0.4194563256837087)), add(abs(neg(-2.3067548438201153)), add(cos(add(x, x)), add(x, x))))))), x))), add(add(cos(add(add(add(x, x), add(x, x)), protectedSqrt(x))), add(add(x, x), x)), add(cos(add(x, x)), add(cos(add(x, x)), add(add(x, x), add(cos(x), add(cos(add(add(protectedSqrt(x), x), cos(add(x, x)))), add(x, add(protectedSqrt(x), add(cos(add(x, x)), x)))))))))))), add(sin(add(cos(add(add(add(x, x), add(sin(add(x, add(x, -0.4194563256837087))), x)), add(add(x, x), x))), x)), add(sin(x), add(add(cos(add(x, x)), add(cos(add(add(cos(add(x, protectedSqrt(x))), add(cos(add(add(x, -0.4194563256837087), x)), x)), -2.20980308243909)), add(add(cos(add(cos(add(add(x, -0.4194563256837087), x)), add(x, x))), add(cos(add(x, add(cos(add(x, add(x, -0.4194563256837087))), x))), neg(-1.899702710288039))), add(cos(add(add(add(x, x), add(x, add(x, -0.4194563256837087))), protectedSqrt(x))), add(x, x))))), add(cos(add(x, x)), add(cos(add(add(add(x, x), add(x, add(x, -0.4194563256837087))), protectedSqrt(x))), add(cos(add(cos(add(add(cos(add(cos(add(x, x)), add(-1.0998860715047434, x))), x), x)), add(x, x))), x)))))))"
tree = gp.PrimitiveTree.from_string(solution, pset)

fn  = gp.compile(tree, pset)
bounds = [(0., 5.)]

param_values = []
for param_bound in bounds:
    param_values += [list(np.linspace(param_bound[0], param_bound[1], num = len(y)))]

points = list(it.product(*param_values))

print(f"Error medio de la solución: {mse(fn, points, y)}")
delta = points[len(points) - 1][0] - points[len(points) - 2][0]
print(f"Valor 427 (t = {5. + delta}):  {fn(5. + delta)}")
print(f"Valor 428 (t = {5. + 2*delta}):  {fn(5. + 2*delta)}")
print(f"Número de nodos: {len(tree)}")