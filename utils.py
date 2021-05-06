import numpy as np

def average_list_of_lists(li):
    li.sort(key = len)

    li_avg = np.zeros(len(li[-1]))
    li_avg_mat = np.zeros((len(li), len(li[-1])))
    length_i_minus_1 = 0
    for i in range(len(li)):
        length_i = len(li[i])
        li_avg_mat[i, 0 : length_i] = np.array(li[i])

    for i in range(len(li) - 1):
        length_i = len(li[i])
        if length_i > length_i_minus_1:
            li_avg[length_i_minus_1: length_i] = np.average(np.array(li_avg_mat[i:, length_i_minus_1: length_i]), axis=0)
            length_i_minus_1 = length_i

    if length_i_minus_1 < len(li[-1]):
        li_avg[length_i_minus_1: len(li[-1])] = np.array(li_avg_mat[len(li) - 1, length_i_minus_1: len(li[-1])])
    return li_avg