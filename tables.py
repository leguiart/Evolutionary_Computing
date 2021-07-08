from tabulate import tabulate
from hklearn_genetic.problem import BinaryRastrigin, BinaryBeale, BinaryHimmelblau, BinaryEggholder
from texttable import Texttable
import latextable

rast = BinaryRastrigin(n_dim = 2, n_prec=8)
beale = BinaryBeale(n_prec=8)
himme = BinaryHimmelblau(n_prec=8)
egg = BinaryEggholder(n_prec=4)
params_bin = {'PS_BINARY': {'Rastrigin': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'proportional'}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'proportional'}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.0125, 'max_iter': 1000, 'selection': 'proportional'}, 'Eggholder': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.020833333333333332, 'max_iter': 1000, 'selection': 'proportional'}}, 'PS_E_BINARY': {'Rastrigin': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.1}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.3}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.3}, 'Eggholder': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.020833333333333332, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.1}}, 'TS_BINARY': {'Rastrigin': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.004166666666666667, 'max_iter': 1000, 'selection': 'tournament'}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'tournament'}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.0125, 'max_iter': 1000, 'selection': 'tournament'}, 'Eggholder': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.015625, 'max_iter': 1000, 'selection': 'tournament'}}, 'TS_E_BINARY': {'Rastrigin': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.0125, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.2}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.0125, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.1}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.1}, 'Eggholder': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.015625, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.3}}, 'SUS_BINARY': {'Rastrigin': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'sus'}, 'Beale': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'sus'}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.004166666666666667, 'max_iter': 1000, 'selection': 'sus'}, 'Eggholder': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.020833333333333332, 'max_iter': 1000, 'selection': 'sus'}}, 'SUS_E_BINARY': {'Rastrigin': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.004166666666666667, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.1}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.016666666666666666, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.1}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.004166666666666667, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.3}, 'Eggholder': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.005208333333333333, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.1}}}
params_real = {'PS_REAL': {'Rastrigin': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.2, 'max_iter': 1000, 'selection': 'proportional'}, 'Beale': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.2, 'max_iter': 1000, 'selection': 'proportional'}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.2, 'max_iter': 1000, 'selection': 'proportional'}, 'Eggholder': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.25, 'max_iter': 1000, 'selection': 'proportional'}}, 'PS_E_REAL': {'Rastrigin': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.5, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.2}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.2, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.2}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.2, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.2}, 'Eggholder': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.2, 'max_iter': 1000, 'selection': 'proportional', 'elitism': 0.1}}, 'TS_REAL': {'Rastrigin': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.2, 'max_iter': 1000, 'selection': 'tournament'}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.2, 'max_iter': 1000, 'selection': 'tournament'}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.25, 'max_iter': 1000, 'selection': 'tournament'}, 'Eggholder': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.5, 'max_iter': 1000, 'selection': 'tournament'}}, 'TS_E_REAL': {'Rastrigin': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.5, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.2}, 'Beale': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.5, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.1}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.5, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.2}, 'Eggholder': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.5, 'max_iter': 1000, 'selection': 'tournament', 'elitism': 0.1}}, 'SUS_REAL': {'Rastrigin': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.25, 'max_iter': 1000, 'selection': 'sus'}, 'Beale': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.5, 'max_iter': 1000, 'selection': 'sus'}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.25, 'max_iter': 1000, 'selection': 'sus'}, 'Eggholder': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.2, 'max_iter': 1000, 'selection': 'sus'}}, 'SUS_E_REAL': {'Rastrigin': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.5, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.2}, 'Beale': {'n_individuals': 500, 'pc': 0.95, 'pm': 0.2, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.1}, 'Himmelblau': {'n_individuals': 500, 'pc': 0.9, 'pm': 0.25, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.1}, 'Eggholder': {'n_individuals': 500, 'pc': 0.85, 'pm': 0.25, 'max_iter': 1000, 'selection': 'sus', 'elitism': 0.3}}}


# table = Texttable()
# table.set_deco(Texttable.HEADER)
# table.set_cols_dtype(['t',  # text
#                        'i',
#                        'f',
#                        'i',
#                        't'])
#                     #   'f',  # float (decimal)
#                     #   'e',  # float (exponent)
#                     #   'i',  # integer
#                     #   'a']) # automatic
# table.set_cols_align(["l", "r", "r", "r", "l"])

for k, v in params_bin.items():
    cols = ["Function"] + list(params_bin[k]["Rastrigin"].keys())
    for i in range(len(cols)):
        cols[i] = cols[i].replace("_", " ")
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    if len(cols) == 6:

        table.set_cols_dtype(['t',  # text
                        'i',
                        'f',
                        'f',
                        'i',
                        't'])
        table.set_cols_align(["l", "r", "r", "r", "r", "l"])
    elif len(cols) == 7:
        table.set_cols_dtype(['t',  # text
                'i',
                'f',
                'f',
                'i',
                'f',
                't'])
        table.set_cols_align(["l", "r", "r", "r", "r", "r", "l"])
    rows = [cols]
    #table.add_rows(cols)
    for_caption = k.replace("_", " ")
    for k1, v1 in v.items():
        first_column = k1
        row = [first_column]
        for v2 in v1.values():
            row += [v2]
        rows += [row]
    table.add_rows(rows)
    #print(table.draw() + "\n")
    print(latextable.draw_latex(table, caption=f"{for_caption} parameters.", label=for_caption) + "\n")

for k, v in params_real.items():
    cols = ["Function"] + list(params_real[k]["Rastrigin"].keys())
    for i in range(len(cols)):
        cols[i] = cols[i].replace("_", " ")
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    if len(cols) == 6:

        table.set_cols_dtype(['t',  # text
                        'i',
                        'f',
                        'f',
                        'i',
                        't'])
        table.set_cols_align(["l", "r", "r", "r", "r", "l"])
    elif len(cols) == 7:
        table.set_cols_dtype(['t',  # text
                'i',
                'f',
                'f',
                'i',
                'f',
                't'])
        table.set_cols_align(["l", "r", "r", "r", "r", "r", "l"])
    rows = [cols]
    for_caption = k.replace("_", " ")
    #table.add_rows(cols)
    for_caption = k
    for k1, v1 in v.items():
        first_column = k1
        row = [first_column]
        for v2 in v1.values():
            row += [v2]
        rows += [row]
    table.add_rows(rows)
    #print(table.draw() + "\n")
    print(latextable.draw_latex(table, caption=f"{for_caption} parameters.", label=for_caption) + "\n")



# table.add_rows([cols,
#                 ["abcd",    "67",    654,   89,    128.001],
#                 ["efghijk", 67.5434, .654,  89.6,  12800000000000000000000.00023],
#                 ["lmn",     5e-78,   5e-78, 89.4,  .000000000000128],
#                 ["opqrstu", .023,    5e+78, 92.,   12800000000000000000000]])
# print(table.draw() + "\n")
# print(latextable.draw_latex(table, caption="Another table.", label="table:another_table") + "\n")
# print(latextable.draw_latex(table, caption="A table with dropped columns.", label="table:dropped_column_table", drop_columns=['exp', 'int']))


# rows = list(params_bin["PS_BINARY"].keys())

# table_bin = tabulate(params_bin["PS_BINARY"])
# table_real = tabulate(params_real["PS_REAL"])
# print(table_bin)
# print(table_real)