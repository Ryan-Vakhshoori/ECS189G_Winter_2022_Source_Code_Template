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
import numpy as np


class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 3
    learning_rate = 1e-3

    def __init__(self, mName, mDescription,hidden_layers, optimizer, activation_function):
        method.__init__(self, mName, mDescription, hidden_layers, optimizer, activation_function)
        nn.Module.__init__(self)
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.optimizer = optimizer
        #(1,20,10)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layers[1], out_channels=hidden_layers[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[1], out_channels=hidden_layers[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layers[1], out_channels=hidden_layers[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * hidden_layers[1], hidden_layers[2])
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return F.log_softmax(x, dim=1)

    def train(self, X):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        loss_function = nn.CrossEntropyLoss()
        resulting_loss = []
        epochs = []
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            total_loss = 0.0
            res_loss = 0.0
            for i, data in enumerate(X, 0):
                inputs = data['image']
                labels = data['label']
                # print(inputs)
                output = self.forward(inputs)
                loss = loss_function(output, labels)
                res_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if i % 200 == 199:  # print every 2000 mini-batches
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {total_loss / 200:.3f}')
                #     total_loss = 0.0
            resulting_loss.append(res_loss / len(X))
            epochs.append(epoch + 1)
            print(f'[{epoch + 1}], loss: {res_loss / len(X):.3f}')
        return resulting_loss, epochs

    def test(self, test_data):
        total = 0
        correct = 0
        predicted_labels = np.array([])
        actual_labels = np.array([])
        for data in test_data:
            inputs = data['image']
            labels = data['label']
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels = np.append(predicted_labels, predicted.numpy())
            actual_labels = np.append(actual_labels, labels.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        # print(f'Test Accuracy: {accuracy}')
        return predicted_labels,actual_labels

    def run(self):
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        resulting_loss,epochs = self.train(self.data['train_data'])
        print('--start testing...')
        predicted_labels, actual_labels = self.test(self.data['test_data'])
        return resulting_loss, epochs, predicted_labels, actual_labels
