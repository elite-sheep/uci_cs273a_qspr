# Copyright @yucwang 2022

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from metrics import rmse, aare
from onehot import gen_numpy
import os
import sys

class FFANN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, weights=None):
        super(FFANN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 2 * hidden_layer_size, bias=True)
        self.activation = nn.Tanh()
        self.hidden_layer2 = nn.Linear(2 * hidden_layer_size, hidden_layer_size, bias=True)
        self.out_layer = nn.Linear(hidden_layer_size, 1, bias = True)
        if weights is not None:
            with torch.no_grad():
                self.hidden_layer.weight.copy_(torch.tensor(weights['hidden_layer'], dtype=torch.float))
                self.hidden_layer.bias.copy_(torch.tensor(weights['hidden_layer_bias'], dtype=torch.float))
                self.out_layer.weight.copy_(torch.tensor(weights['output_layer'], dtype=torch.float))
                self.out_layer.bias.copy_(torch.tensor(weights['output_layer_bias'], dtype=torch.float))

    def init(self):
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer2.weight)
        torch.nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, x):
        hidden = self.hidden_layer(x)
        act = self.activation(hidden)
        hidden2 = self.hidden_layer2(act)
        act2 = self.activation(hidden2)
        y = self.out_layer(act2)
        return y

    def save(self, path):
        torch.save(self.state_dict(), path)

def train(train_x, train_y, hidden_layer_size, lr=0.001, n_epoches=8, out_dir="./weights/"):
    num_of_data_points = train_x.shape[0]
    input_size = train_x.shape[1]
    train_y = train_y.view(train_y.shape[0], 1)
    learner = FFANN(input_size, hidden_layer_size, weights=None)
    #learner.load_state_dict(torch.load('./weights/baseline_weights.pt'))
    #learner.eval()
    learner.init()
    loss_func = nn.L1Loss()
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4096,16384,65536], gamma=0.4)

    batch_size = int(num_of_data_points / 8 + 1)
    for epoch in range(n_epoches):
        for i in range(8):
            running_loss = 0.0
            optimizer.zero_grad()
            y = learner(train_x[i*batch_size:(i+1)*batch_size])
            loss = loss_func(y, train_y[i*batch_size:(i+1)*batch_size])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()
        if epoch % 16 == 0:
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

def test(input_x, hidden_layer_size, weights_file, ref_y=None):
    num_of_data_points = input_x.shape[0]
    input_size = input_x.shape[1]

    # Initialize the neural network
    learner = FFANN(input_size, hidden_layer_size)
    learner.load_state_dict(torch.load(weights_file))
    learner.eval()

    y = np.zeros(num_of_data_points)
    for i in range(num_of_data_points):
        y[i] = learner(input_x[i])

    y_rmse = None
    y_aare = None
    if ref_y is not None:
        y_rmse = rmse(y, ref_y)
        y_aare = aare(y, ref_y)

    return y, y_rmse, y_aare

# weights = np.genfromtxt("./weights/ffann4_weights.txt", delimiter=None)
# weights = {
#         "hidden_layer": weights[0:72,:].T,
#         "hidden_layer_bias": weights[72,:],
#         "output_layer": weights[-2,:],
#         "output_layer_bias": weights[-1,0]
#         }
# 
# network = FFANN(72, 4, weights)
# network.save("./weights/baseline_weights.pt")
# 
# test_x, test_y = gen_numpy("./testx.txt", "./testy.txt", "./n_max_min.txt")


#test_x = np.genfromtxt("./testx.txt", delimiter=None)
#test_y = np.genfromtxt("./testy.txt", delimiter=None)

#n_max_min = np.genfromtxt("./n_max_min.txt", delimiter=None)

#n_max = n_max_min[:,1]
#n_min = n_max_min[:,0]
#print(n_max)
#for i in range(test_x.shape[0]):
#    test_x[i] = 2. * (test_x[i] - n_min) / (n_max - n_min) - 1.

# y, y_rmse, y_aare = test(torch.tensor(test_x, dtype=torch.float), 4, "./weights/baseline_weights.pt", test_y)
min_y = 0.3985
max_y = 11.230
#min_y = np.min(y)
#max_y = np.max(y)
# y = 0.5 * (y + 1.) * (max_y - min_y) + min_y
# test_y = 0.5 * (test_y + 1.) * (max_y - min_y) + min_y
#print(y)
#print(test_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFANN for QSPRs')
    phase = sys.argv[2]
    if phase == 'train':
        parser.add_argument('--phase', type=str, help='The phase.')
        parser.add_argument('--trainx', type=str, help='The x for training.')
        parser.add_argument('--trainy', type=str, help='The y for training')
        parser.add_argument('--minmax', type=str, help='The minmax for test')
        parser.add_argument('--outdir', type=str, help='The outdir')
        parser.add_argument('--input_size', type=int, help='The size of input layer')
        parser.add_argument('--hidden_layer_size', type=int, help='The size of hidden_layer')
        args = parser.parse_args()
        train_x, train_y = gen_numpy(args.trainx, args.trainy, args.minmax)
        _ = train(torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.float),
                args.hidden_layer_size, out_dir=args.outdir, n_epoches=32769)
    elif phase == 'test':
        parser.add_argument('--phase', type=str, help='The phase.')
        parser.add_argument('--testx', type=str, help='The x for test.')
        parser.add_argument('--testy', type=str, help='The y for test')
        parser.add_argument('--minmax', type=str, help='The minmax for test')
        parser.add_argument('--weights_file', type=str, help='The weights file')
        parser.add_argument('--input_size', type=int, help='The size of input layer')
        parser.add_argument('--hidden_layer_size', type=int, help='The size of hidden_layer')
        args = parser.parse_args()
        test_x, test_y = gen_numpy(args.testx, args.testy, args.minmax)
        y, y_rmse, y_aare = test(torch.tensor(test_x, dtype=torch.float), args.hidden_layer_size, args.weights_file, test_y)
        y = 0.5 * (y + 1.) * (max_y - min_y) + min_y
        test_y = 0.5 * (test_y + 1.) * (max_y - min_y) + min_y
        y_rmse = rmse(y, test_y)
        print(y_rmse)
        print(y_aare)
