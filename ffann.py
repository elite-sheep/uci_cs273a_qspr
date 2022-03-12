# Copyright @yucwang 2022

import torch
import torch.nn as nn

class FFANN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, weights=None):
        super(FFANN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_layer_size, bias=False)
        self.out_layer = nn.Linear(hidden_layer_size, 1, bias = False)
        if weights is not None:
            self.hidden_layer.weight = weights['hidden_layer']
            self.out_layer = weights['output_layer']

    def forward(self, x):
        hidden = self.hidden_layer(x)
        y = self.out_layer(x)
        return y
