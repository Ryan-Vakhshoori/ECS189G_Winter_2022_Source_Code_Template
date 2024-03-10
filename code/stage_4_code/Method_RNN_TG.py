from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.init as init

import numpy as np

# TODO: THIS FILE IS JUST COPIED FORM METHOD_RNN_TC

class Method_RNN_TG(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, hidden_size, num_layers, optimizer, activation_function, word_to_one_hot, one_hot_to_word):
        method.__init__(self, mName, mDescription, hidden_size, optimizer, activation_function)
        nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.optimizer = optimizer
        if mName == "RNN":
            print("RNN Method")
            self.rnn = nn.RNN(input_size=6478, hidden_size=self.hidden_size, num_layers=2, batch_first=True)

        elif mName == "LSTM":
            print("LSTM Method")
            self.rnn = nn.LSTM(input_size=6478, hidden_size=self.hidden_size, num_layers=2, batch_first=True)

        elif mName == "GRU":
            print("GRU Method")
            self.rnn = nn.GRU(input_size=6478, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        #self.rnn = nn.LSTM(input_size=6478, hidden_size=self.hidden_size, num_layers=2, batch_first=True) # try 100 with more epochs
        self.fc = nn.Linear(self.hidden_size, 6478)
        self.word_to_one_hot = word_to_one_hot
        self.one_hot_to_word = one_hot_to_word

        # for name, param in self.rnn.named_parameters():
        #     if 'weight' in name:
        #         init.orthogonal_(param)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:,-1,:]
        output = self.fc(output)
        return output

    def train(self, X):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()
        resulting_loss = []
        epochs = []
        for epoch in range(self.max_epoch):
            total_loss = 0.0
            res_loss = 0.0

            for i, data in enumerate(X, 0):
                inputs = data['input']
                target = data['target']
                # print(inputs,target)

                output = self.forward(inputs)

                # target = target.view(-1)

                loss = loss_function(output,target)

                res_loss += loss.item()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                optimizer.step()
            resulting_loss.append(res_loss / len(X))
            epochs.append(epoch)
            print(f'[{epoch + 1}], loss: {res_loss / len(X):.3f}')




            # for i, data in enumerate(X, 0):
            #     inputs = data['text']
            #     labels = data['label']
            #     # print(f"Print the shape of the input batch: {inputs.shape}")
            #     output = self.forward(inputs)
            #     loss = loss_function(output.view(-1,6477,))
            #     # total_loss += loss.item()
            #     res_loss += loss.item()
            #     optimizer.zero_grad()
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            # resulting_loss.append(res_loss / len(X))
            # epochs.append(epoch)
            # print(f'[{epoch + 1}], loss: {res_loss / len(X):.3f}')
        return resulting_loss, epochs


    def predict(self,input_string):
        input_string = input_string.lower().split()  # Splitting without removing punctuation
        input_string = [self.word_to_one_hot[token] for token in input_string]

        input = torch.stack(input_string).unsqueeze(0)

        loop = 10

        for i in range(loop):
            pred = self.predict_next_word(input)
            input_string.append(pred)
            input = torch.stack(input_string).unsqueeze(0)


        output = []
        for token in input_string:
            output.append(self.one_hot_to_word[tuple(token.numpy().tolist())])
        # print(self.one_hot_to_word[tuple(one_hot.numpy().tolist())])
        print(" ".join(output))
        return output
        # with torch.no_grad():
        #     output = self.forward(input)



    def predict_next_word(self, input):
        # self.eval()
       #
       #  input_string = input_string.lower().split()  # Splitting without removing punctuation
       #  input_string = [self.word_to_one_hot[token] for token in input_string]
       #
       #  print(input_string)
       # # print(input_string.shape())
       #
       #  input = torch.stack(input_string).unsqueeze(0)
       #  print(input.size())
       #
       #
       #
       #  # convert input to input data


        with torch.no_grad():
            output = self.forward(input)

        # print("output word:")
        # print(output)

        softmaxed_tensor = torch.nn.functional.softmax(output, dim=1)

        # print(softmaxed_tensor)

        # Perform one-hot encoding
        _, one_hot_encoded = torch.max(softmaxed_tensor, 1)

        # print(one_hot_encoded)

        one_hot = torch.zeros(6478)
        one_hot[one_hot_encoded[0]] = 1

        # print(self.one_hot_to_word[tuple(one_hot.numpy().tolist())])
        return one_hot

    # def test(self, test_data):
    #     total = 0
    #     correct = 0
    #     predicted_labels = np.array([])
    #     actual_labels = np.array([])
    #     with torch.no_grad():
    #         for data in test_data:
    #             inputs = data['text']
    #             labels = data['label']
    #             outputs = self.forward(inputs)
    #             predicted = torch.tensor([1 if i == True else 0 for i in outputs > 0.5])
    #             predicted_labels = np.append(predicted_labels, predicted.numpy())
    #             actual_labels = np.append(actual_labels, labels.numpy())
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #
    #     accuracy = correct / total
    #     print(f'Test Accuracy: {accuracy}')
    #     return predicted_labels, actual_labels

    def run(self):
        print('method running...')
        print('--start training...')
        resulting_loss, epochs = self.train(self.data)
        print('--end training...')

        self.predict("What did the")
        self.predict("I am going")

        # print('--start testing...')
        # predicted_labels, actual_labels = self.test(self.data['test_data'])
        # return resulting_loss, epochs, predicted_labels, actual_labels
        return resulting_loss, epochs
