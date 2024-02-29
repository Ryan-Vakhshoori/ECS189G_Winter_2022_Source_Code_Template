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
    max_epoch = 10
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, hidden_size, num_layers, optimizer, activation_function):
        method.__init__(self, mName, mDescription, hidden_size, optimizer, activation_function)
        nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.optimizer = optimizer
        # epochs 20
        # batch size 16
        # self.rnn = nn.RNN(input_size=50, hidden_size=200, num_layers=2, batch_first=True)
        # self.rnn = nn.GRU(input_size=50, hidden_size=200, num_layers=1, batch_first=True)
        # self.rnn = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.82548
        # self.rnn = nn.LSTM(input_size=50, hidden_size=100, num_layers=1, batch_first=True) 0.81112
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.82404
        # batch size 64
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.84144
        # self.rnn = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.81344
        # self.rnn = nn.GRU(input_size=50, hidden_size=100, num_layers=1, batch_first=True) 0.8316
        # epochs 25
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.83324
        # self.rnn = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.83076
        # epochs 15
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.84468
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.8496
        # max_len 120 only next one
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.83212
        # max_len 122 only next one
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.84204
        # max_len 130 only next one
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.84964
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True, dropout=0.1) 0.84588
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2) 0.84932
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True, dropout=0.3) 0.83712
        # 10 epochs
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True)  0.8438
        # 14 epochs
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.84908
        # 16 epochs
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.84864
        # max_len 135 only next one
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.84836
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.84744
        # self.rnn = nn.GRU(input_size=50, hidden_size=45, num_layers=2, batch_first=True) 0.84372
        # self.rnn = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.62548
        # self.rnn = nn.GRU(input_size=50, hidden_size=55, num_layers=2, batch_first=True) 0.84524
        # epochs 16
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.83744
        # epochs 10
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.82392
        # epochs 12
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True) 0.8438
        # epochs 14
        # self.rnn = nn.GRU(input_size=50, hidden_size=50, num_layers=2, batch_first=True) 0.83696
        # self.rnn = nn.GRU(input_size=50, hidden_size=55, num_layers=2, batch_first=True) 0.84104
        # 15 epochs
        # self.rnn = nn.GRU(input_size=50, hidden_size=100, num_layers=2, batch_first=True) 0.834
        # self.rnn = nn.GRU(input_size=50, hidden_size=100, num_layers=2, batch_first=True) 0.8242
        # self.rnn = nn.GRU(input_size=50, hidden_size=200, num_layers=2, batch_first=True, dropout=0.2) 0.83852
        # self.rnn = nn.LSTM(input_size=50, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        # self.rnn = nn.GRU(input_size=50, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2) 0.83072
        self.rnn = nn.GRU(input_size=100, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
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
            res_loss = 0.0
            for i, data in enumerate(X, 0):
                inputs = data['embedding']
                labels = data['label']
                output = self.forward(inputs)
                loss = loss_function(output.squeeze(), labels.float())
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
