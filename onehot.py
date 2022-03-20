# Copyright @yucwang 2022

import numpy as np

min_y = 0.3985
max_y = 11.230

def gen_numpy(x_file, y_file, max_min_file):
    x = np.genfromtxt(x_file, delimiter=None)
    y = np.genfromtxt(y_file, delimiter=None)
    n_max_min = np.genfromtxt(max_min_file, delimiter=None)

    n_min = n_max_min[:,0]
    n_max = n_max_min[:,1]
    x[:,] = 2. * (x[:,] - n_min) / (n_max - n_min) - 1.
    y = 2. * (y - min_y) / (max_y - min_y) - 1.

    return x, y
