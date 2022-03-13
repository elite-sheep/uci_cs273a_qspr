# Copyright @yucwang 2022

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from metrics import rmse, aare
import os

class FFANN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, weights=None):
        super(FFANN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_layer_size, bias=True)
        self.out_layer = nn.Linear(hidden_layer_size, 1, bias = True)
        if weights is not None:
            with torch.no_grad():
                self.hidden_layer.weight.copy_(torch.tensor(weights['hidden_layer'], dtype=torch.float))
                self.hidden_layer.bias.copy_(torch.tensor(weights['hidden_layer_bias'], dtype=torch.float))
                self.out_layer.weight.copy_(torch.tensor(weights['output_layer'], dtype=torch.float))
                self.out_layer.bias.copy_(torch.tensor(weights['output_layer_bias'], dtype=torch.float))

    def forward(self, x):
        hidden = self.hidden_layer(x)
        y = self.out_layer(hidden)
        return y

def train(train_x, train_y, hidden_layer_size, lr=0.01, momentum=0.9, n_epoches=8, out_dir="./weights/"):
    num_of_data_points = train_x.shape[0]
    input_size = train_x.shape[1]
    learner = FFANN(input_size, hidden_layer_size, weights=None)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(learner.parameters(), lr=lr, momentum=momentum)

    for epoch in range(n_epoches):
        running_loss = 0.0
        for i, x in enumerate(train_x):
            y_ref = train_y[i]
            optimizer.zero_grad()
            y = learner(x)
            loss = loss_func(y, y_ref)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        torch.save(learner.state_dict(), os.path.join(out_dir, "weights_{}.pt".format(epoch)))
        print("Epoch #{}: Loss = {}".format(epoch, running_loss/num_of_data_points))

    return learner

def test(input_x, hidden_layer_size, weights, ref_y=None):
    num_of_data_points = input_x.shape[0]
    input_size = input_x.shape[1]

    # Initialize the neural network
    learner = FFANN(input_size, hidden_layer_size, weights)

    y = np.zeros(num_of_data_points)
    for i in range(num_of_data_points):
        y[i] = learner(input_x[i])

    y_rmse = None
    y_aare = None
    if ref_y is not None:
        y_rmse = rmse(y, ref_y)
        y_aare = aare(y, ref_y)

    return y, y_rmse, y_aare

weights = np.genfromtxt("./weights/ffann4_weights.txt", delimiter=None)
weights = {
        "hidden_layer": weights[0:72,:].T,
        "hidden_layer_bias": weights[72,:],
        "output_layer": weights[-2,:],
        "output_layer_bias": weights[-1,0]
        }
network = FFANN(72, 4, weights=weights)

test_x = np.genfromtxt("./testx.txt", delimiter=None)
test_y = np.genfromtxt("./testy.txt", delimiter=None)

n_max_min = np.genfromtxt("./n_max_min.txt", delimiter=None)

n_max = n_max_min[:,1]
n_min = n_max_min[:,0]
print(n_max)
for i in range(test_x.shape[0]):
    test_x[i] = 2. * (test_x[i] - n_min) / (n_max - n_min) - 1.

y, y_rmse, _ = test(torch.tensor(test_x, dtype=torch.float), 4, weights, test_y)
min_y = 0.3985
max_y = 11.230
y = 0.5 * (y + 1.) * (max_y - min_y) + min_y
print(y)
print(test_y)
y_rmse = rmse(y, test_y)
print(y-test_y)
