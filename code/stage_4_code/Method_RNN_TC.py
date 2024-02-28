'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np



class Method_RNN_TC(method, nn.Module):
    data = None
    max_epoch = 5
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, hidden_size, num_layers, optimizer, activation_function):
        method.__init__(self, mName, mDescription, hidden_size, optimizer, activation_function)
        nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.rnn = nn.RNN(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output

    def train(self, X):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loss_function = nn.BCELoss()
        print(self.optimizer)
        resulting_loss = []
        epochs = []
        for epoch in range(self.max_epoch):
            total_loss = 0.0
            res_loss = 0.0
            for i, (inputs, labels) in enumerate(X, 0):
                # inputs = data['text']
                # labels = data['label']
                # print(f"Print the shape of the input batch: {inputs.shape}")
                output = self.forward(inputs)
                loss = loss_function(output.squeeze(), labels.float())
                # total_loss += loss.item()
                res_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            resulting_loss.append(res_loss / len(X))
            epochs.append(epoch)
            print(f'[{epoch + 1}], loss: {res_loss / len(X):.3f}')

        return resulting_loss, epochs

    def test(self, test_data):
        total = 0
        correct = 0
        predicted_labels = np.array([])
        actual_labels = np.array([])
        with torch.no_grad():
            for inputs, labels in test_data:
                # inputs = data['text']
                # labels = data['label']
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
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        resulting_loss, epochs = self.train(self.data['train_data'])
        print('--start testing...')
        predicted_labels, actual_labels = self.test(self.data['test_data'])
        return resulting_loss, epochs, predicted_labels, actual_labels
