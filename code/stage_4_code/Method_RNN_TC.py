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
from transformers import BertTokenizer, BertModel


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

        self.embedding = nn.Embedding(len(self.data['TEXT'].vocab), self.hidden_size)
        self.rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output

    def train(self, X):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()
        print(self.optimizer)
        resulting_loss = []
        epochs = []
        for epoch in range(self.max_epoch):
            total_loss = 0.0
            res_loss = 0.0
            for i, data in enumerate(X, 0):
                inputs = data['text']
                labels = data['label']
                tokens = tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, return_tensors='pt',
                                                     add_special_tokens=True)
                inputs = model(tokens['inputs_ids'], attention_mask=tokens['attention_mask'])
                inputs = inputs.last_hidden_state
                print(f"Print the shape of the input batch: {inputs.shape}")
                output = self.forward(inputs)
                loss = loss_function(output, labels)
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
            for data in test_data:
                inputs = data.text[0]
                labels = data.label
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted_labels = np.append(predicted_labels, predicted.numpy())
                actual_labels = np.append(actual_labels, labels.numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return predicted_labels, actual_labels

    def run(self):
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        resulting_loss, epochs = self.train(self.data['train_data'])
        print('--start testing...')
        predicted_labels, actual_labels = self.test(self.data['test_data'])
        return resulting_loss, epochs, predicted_labels, actual_labels

