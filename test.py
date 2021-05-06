import numpy as np
from utils import average_list_of_lists

def generate_random_lists(length, range_floats, range_len):
    li = []
    for i in range(length):
        list_length = np.random.randint(range_len[0], high = range_len[1])
        li += [list(np.random.uniform(range_floats[0], high = range_floats[1], size = list_length))]
    return li

li = generate_random_lists(5, (1, 100), (10, 50))

for l in li:
    print(len(l))

li.sort(key = len)
li += [list(np.random.uniform(1, high = 100, size = len(li[-1])))]
print(li)

li_avg = average_list_of_lists(li)


print(len(li_avg))
print(li_avg)