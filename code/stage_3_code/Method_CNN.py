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


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 3
    learning_rate = 1e-3

    def __init__(self, mName, mDescription,hidden_layers, optimizer, activation_function):
        method.__init__(self, mName, mDescription, hidden_layers, optimizer, activation_function)
        nn.Module.__init__(self)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.final_layer = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        output = self.layer_1(x)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.flatten(output)
        output = self.final_layer(output)
        return output

    def train(self, X):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        resulting_loss = []
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            total_loss = 0.0
            res_loss = 0.0
            for i, data in enumerate(X, 0):
                inputs = data['image']
                labels = data['label']
                output = self.forward(inputs)
                loss = loss_function(output, labels)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 200 == 0:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {total_loss / 200:.3f}')
                    res_loss += total_loss
                    total_loss = 0.0
            resulting_loss.append(res_loss / len(X))
        return resulting_loss

    def test(self, test_data):
        predict,labels = [],[]
        total = 0
        correct = 0
        for data in test_data:
            inputs = data['image']
            labels = data['label']
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
<<<<<<< HEAD
            total += label.size(0)
            correct += (predicted == label).sum().item()
        accuracy = correct / total
        return accuracy
=======
            print(predicted)
            print(labels)
            predict.append(predicted)
            labels.append(label)
        return {'predict': predict, 'labels': labels}
>>>>>>> cd7ec2ba58dfc94adf1811f95677ff5598aee0f8

    def run(self):
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        resulting_loss = self.train(self.data['train_data'])
        print('--start testing...')
        accuracy_ev = self.test(self.data['test_data'])
        return {'resulting_loss': resulting_loss, 'epoch': self.max_epoch, 'accuracy': accuracy_ev}
