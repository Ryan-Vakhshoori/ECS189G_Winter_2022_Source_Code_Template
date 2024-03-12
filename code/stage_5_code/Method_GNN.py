'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_5_code.Evaluation_Metrics import Evaluation_Metrics
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from code.stage_5_code.Layers import GraphConvolution

class Method_GNN(method, nn.Module):
    data = None
    max_epoch = 35
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, hidden_size, num_layers, optimizer, activation_function):
        method.__init__(self, mName, mDescription, hidden_size, optimizer, activation_function)
        nn.Module.__init__(self)

        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.optimizer = optimizer
        self.activation_function = activation_function
        self.hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.gc1 = GraphConvolution(1433, 300)
        self.gc2 = GraphConvolution(300, 7)
        self.dropout = 0.5

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def train(self, features, labels, adj, idx_train, idx_val):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        print(self.optimizer)
        resulting_loss = []
        epochs = []
        for epoch in range(self.max_epoch):
            output = self.forward(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            epochs.append(epoch)
            resulting_loss.append(loss_train.item())
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()))
        return resulting_loss, epochs

    def test(self, features, adj, labels, idx_test):
        output = self.forward(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        # acc_test = accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return output[idx_test], labels[idx_test]

    def run(self):
        # accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        train_idx = self.data['train_test_val']['idx_train']
        features = self.data['graph']['X']
        labels = self.data['graph']['y']
        adj = self.data['graph']['utility']['A']
        val_idx = self.data['train_test_val']['idx_val']
        test_idx = self.data['train_test_val']['idx_test']
        resulting_loss, epochs = self.train(features, labels, adj, train_idx, val_idx)
        print('--start testing...')
        output, labels = self.test(features, adj, labels, test_idx)
        return resulting_loss, epochs, output, labels