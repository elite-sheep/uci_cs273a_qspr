# Copyright @yucwang 2022

import torch
import torch.nn as nn
import numpy as np
from metrics import log_rmse, aare

class FFANN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, weights=None):
        super(FFANN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_layer_size, bias=False)
        self.out_layer = nn.Linear(hidden_layer_size, 1, bias = False)
        if weights is not None:
            with torch.no_grad():
                self.hidden_layer.weight.copy_(torch.tensor(weights['hidden_layer']))
                self.out_layer.weight.copy_(torch.tensor(weights['output_layer']))

    def forward(self, x):
        hidden = self.hidden_layer(x)
        y = self.out_layer(x)
        return y

def test(input_x, hidden_layer_size, weights, ref_y=None):
    num_of_data_points = input_x.shape[0]
    input_size = input_x.shape[1]

    # Initialize the neural network
    learner = FFANN(input_size, hidden_layer_size, weights)

    y = np.zeros(num_of_data_points)
    for i in range(num_of_data_points):
        y[i] = learner(input_x[i])

    log_rmse = None
    aare = None
    if ref_y is not None:
        log_rmse = log_rmse(y, ref_y)
        aare = aare(y, ref_y)

    return y

weights = np.genfromtxt("./weights/ffann4_weights.txt", delimiter=None)
weights = {
        "hidden_layer": weights[0:72,:].T,
        "output_layer": weights[-1,:].T
        }
network = FFANN(72, 4, weights=weights)
