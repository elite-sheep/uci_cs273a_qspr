# Copyright @yucwang 2022

import numpy as np

def rmse(y, y_reference):
    mse = np.square(y - y_reference).mean()
    return np.sqrt(mse)

def log_rmse(y, y_reference):
    log_diff = np.log(np.divide(y, y_reference))
    log_mse = np.square(log_diff).mean()
    return np.sqrt(log_mse)

def aare(y, y_reference):
    relative_diff = np.abs(np.divide(y, y_reference) - 1.)
    return relative_diff.mean()
