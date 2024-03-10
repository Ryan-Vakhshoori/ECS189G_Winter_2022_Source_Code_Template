'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_function):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation_function = activation_function

    def forward(self, x, adj):
        x = self.fc(x)
        x = torch.mm(adj, x)
        x = self.activation_function(x)
        return x


class Method_GNN(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3

    def __init__(self):
        method.__init__(self)
        nn.Module.__init__(self)

    def run(self):
        # accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        resulting_loss, epochs = self.train(self.data['train_data'])
        print('--start testing...')
        predicted_labels, actual_labels = self.test(self.data['test_data'])
        return resulting_loss, epochs, predicted_labels, actual_labels