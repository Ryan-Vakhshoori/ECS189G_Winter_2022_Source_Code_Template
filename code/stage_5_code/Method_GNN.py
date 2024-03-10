'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCNLayer(nn.Module):
    def __init__(self, mName, mDescription, hidden_size, num_layers, optimizer, activation_functione):
        super(GCNLayer, self).__init__()

        self.gc1 = GraphConvolution(input_dim, hidden)
        self.gc2 = GraphConvolution(mName, mDescription, hidden_size, num_layers, optimizer, activation_function)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class Method_GNN(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3

    def __init__(self, input_size, hidden_size, output_size):
        method.__init__(self, input_size, hidden_size, output_size)
        nn.Module.__init__(self)

        self.gcn1 = GCNLayer()
        self.gcn2 = GCNLayer(hidden_dim, output_dim, activation_function)

    def test(self, test_data):
        total = 0
        correct = 0
        predicted_labels = np.array([])
        actual_labels = np.array([])
        with torch.no_grad():
            for data in test_data:
                inputs = data['embedding']
                labels = data['label']
                outputs = self.forward(inputs)
                predicted = torch.tensor([1 if i == True else 0 for i in outputs > 0.5])
                predicted_labels = np.append(predicted_labels, predicted.numpy())
                actual_labels = np.append(actual_labels, labels.numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')
        return predicted_labels, actual_labels

    def run(self):
        # accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        resulting_loss, epochs = self.train(self.data['train_data'])
        print('--start testing...')
        predicted_labels, actual_labels = self.test(self.data['test_data'])
        return resulting_loss, epochs, predicted_labels, actual_labels