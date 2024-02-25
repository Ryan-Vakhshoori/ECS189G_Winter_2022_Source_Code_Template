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


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3

    def __init__(self, mName, mDescription,hidden_layers, optimizer, activation_function):
        method.__init__(self, mName, mDescription, hidden_layers, optimizer, activation_function)
        nn.Module.__init__(self)

        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.optimizer = optimizer
        #print(self.hidden_layers)
        # (32, 64, 128, 256, 1024, 512)
        self.rnn = nn.RNN()


    def forward(self, x):
        output = self.layer_1(x)
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
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
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
                # if i % 200 == 199:  # print every 2000 mini-batches
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {total_loss / 200:.3f}')
                #     total_loss = 0.0
            resulting_loss.append(res_loss / len(X))
            epochs.append(epoch)
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
        return predicted_labels, actual_labels

    def run(self):
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        print('method running...')
        print('--start training...')
        resulting_loss, epochs = self.train(self.data['train_data'])
        print('--start testing...')
        predicted_labels, actual_labels = self.test(self.data['test_data'])
        return resulting_loss, epochs, predicted_labels, actual_labels

