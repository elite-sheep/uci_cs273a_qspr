# Copyright @yucwang 2022

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from metrics import rmse, aare
from onehot import gen_numpy
from deepchem import gen_numpy_deepchem
import os
import sys

def train(train_x, train_y, hidden_layer_size, lr=0.01, n_epoches=8, out_dir="./weights/"):
    num_of_data_points = train_x.shape[0]
    input_size = train_x.shape[1]
    train_y = train_y.view(train_y.shape[0], 1, 1)
    train_x = train_x.reshape((num_of_data_points, 1, train_x.shape[1]))
    learner = nn.RNN(input_size, hidden_size=hidden_layer_size, num_layers=2)
    #learner.load_state_dict(torch.load('./weights/baseline_weights.pt'))
    #learner.eval()
    #learner.init()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32768,65536], gamma=0.1)

    batch_size = int(num_of_data_points / 8 + 1)
    for epoch in range(n_epoches):
        for i in range(8):
            idx = torch.randint(0, num_of_data_points, (batch_size,))
            running_loss = 0.0
            optimizer.zero_grad()
            y,hn = learner(train_x[idx])
            loss = loss_func(y, train_y[idx])
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
    learner = nn.RNN(input_size, hidden_size=hidden_layer_size, num_layers=2)

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

    input_x = input_x.reshape((num_of_data_points, 1, input_x.shape[1]))
    # Initialize the neural network
    learner = nn.RNN(input_size, hidden_size=hidden_layer_size, num_layers=2)
    learner.load_state_dict(torch.load(weights_file))
    learner.eval()

    y = np.zeros(num_of_data_points)
    for i in range(num_of_data_points):
        y[i], h = learner(input_x[i].view(1, 1, input_x.shape[2]))

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
        train_x, train_y = gen_numpy_deepchem(args.trainx, args.trainy, args.minmax)
        _ = train(torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.float),
                args.hidden_layer_size, out_dir=args.outdir, n_epoches=65537)
    elif phase == 'test':
        parser.add_argument('--phase', type=str, help='The phase.')
        parser.add_argument('--testx', type=str, help='The x for test.')
        parser.add_argument('--testy', type=str, help='The y for test')
        parser.add_argument('--minmax', type=str, help='The minmax for test')
        parser.add_argument('--weights_file', type=str, help='The weights file')
        parser.add_argument('--input_size', type=int, help='The size of input layer')
        parser.add_argument('--hidden_layer_size', type=int, help='The size of hidden_layer')
        args = parser.parse_args()
        test_x, test_y = gen_numpy_deepchem(args.testx, args.testy, args.minmax)
        y, y_rmse, y_aare = test(torch.tensor(test_x, dtype=torch.float), args.hidden_layer_size, args.weights_file, test_y)
        y = 0.5 * (y + 1.) * (max_y - min_y) + min_y
        test_y = 0.5 * (test_y + 1.) * (max_y - min_y) + min_y
        y_rmse = rmse(y, test_y)
        print(y_rmse)
        print(y_aare)
