import string

import torch

from code.base_class.dataset import dataset
import pandas as pd
class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    max_length = None
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    import numpy as np
    import torch
    from torch.nn.functional import one_hot

    # Load jokes dataset from file
    # def load_jokes_dataset(file_path):
    #     jokes = []
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         next(file)  # Skip header
    #         for line in file:
    #             joke = line.strip().split(',')[1]
    #             jokes.append(joke)
    #     return jokes
    #
    # # Jokes dataset file path
    # file_path = 'jokes_dataset.csv'  # Update with the actual file path
    #
    # # Load jokes dataset
    # jokes = load_jokes_dataset(file_path)
    #
    # # Tokenize the jokes and create vocabulary
    # word_to_idx = {}
    # idx_to_word = {}
    # for joke in jokes:
    #     tokens = joke.lower().split()
    #     for token in tokens:
    #         if token not in word_to_idx:
    #             idx = len(word_to_idx)
    #             word_to_idx[token] = idx
    #             idx_to_word[idx] = token
    #
    # # Convert jokes to one-hot encoded tensors
    # def encode_jokes(jokes, word_to_idx):
    #     encoded_jokes = []
    #     for joke in jokes:
    #         tokens = joke.lower().split()
    #         encoded_joke = [word_to_idx[token] for token in tokens if token in word_to_idx]
    #         encoded_jokes.append(encoded_joke)
    #     return encoded_jokes
    #
    # encoded_jokes = encode_jokes(jokes, word_to_idx)
    #
    # # Maximum length of jokes
    # max_length = max(len(joke) for joke in encoded_jokes)
    #
    # # One-hot encode the jokes
    # def one_hot_encode(encoded_jokes, vocab_size, max_length):
    #     one_hot_encode
    # def tokenize_joke(self, joke):
    #     # Tokenize by whitespace and punctuation
    #     joke_tokens = joke.lower().translate(str.maketrans('', '', string.punctuation)).split()
    #     return joke_tokens

    def tokenize_joke(self, joke):
        joke_tokens = joke.lower().split()  # Splitting without removing punctuation
        return joke_tokens


    def load(self):
        file_path = self.dataset_source_folder_path + self.dataset_source_file_name

        df = pd.read_csv(file_path)

        all_tokens = []
        for joke in df['Joke']:
            joke_tokens = self.tokenize_joke(joke)
            all_tokens.extend(joke_tokens)

        vocab = sorted(set(all_tokens))
        vocab_size = len(vocab)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for idx, word in enumerate(vocab)}

        def one_hot_encode(word_idx, vocab_size):
            one_hot = torch.zeros(vocab_size)
            one_hot[word_idx] = 1
            return one_hot


        # Create mapping from word to one-hot encoding
        word_to_one_hot = {word: one_hot_encode(idx, vocab_size) for word, idx in word_to_idx.items()}

        # Create mapping from one-hot encoding to word
        one_hot_to_word = {tuple(one_hot.numpy().tolist()): word for word, one_hot in word_to_one_hot.items()}

        def create_training_data():
            dataX = []
            dataY = []
            jok = 0
            for joke in df['Joke']:
                joke_tokens = self.tokenize_joke(joke)
                joke_length = len(joke_tokens)
                for i in range(joke_length - 1):  # Iterate over the tokens in the joke
                    seq_in = joke_tokens[:i + 1]  # Sequence includes tokens up to the current token

                    seq_out = joke_tokens[i + 1]  # Next token is the output
                    dataX.append([word_to_one_hot[token] for token in seq_in])
                    dataY.append(word_to_one_hot[seq_out])
            n_patterns = len(dataX)
            return dataX, dataY

        training_data_X, training_data_Y = create_training_data()

        word = one_hot_to_word[tuple(training_data_X[0][0].numpy().tolist())]
        print(word, end=" ")
        word2 = one_hot_to_word[tuple(training_data_Y[0].numpy().tolist())]
        print(word2, end=" ")




        return {'train_data': training_data_X, 'test_data': training_data_Y}