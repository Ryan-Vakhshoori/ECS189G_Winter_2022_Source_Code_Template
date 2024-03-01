import string

import torch
import torch.utils.data
from torch.utils.data import Dataset

from code.base_class.dataset import dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.stack(sample['input'])  # Stack list of tensors into a single tensor
        label = sample['target']
        return features, label

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    max_length = None
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

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
        all_tokens.append('<eos>')
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
            data = []
            for joke in df['Joke']:
                joke_tokens = self.tokenize_joke(joke)
                joke_length = len(joke_tokens)
                for i in range(joke_length - 1):  # Iterate over the tokens in the joke
                    seq_in = joke_tokens[:i + 1]  # Sequence includes tokens up to the current token

                    seq_out = joke_tokens[i + 1]  # Next token is the output
                    encoding =[word_to_one_hot[token] for token in seq_in]
                    data.append({'input':torch.stack(encoding), 'target':torch.tensor(word_to_one_hot[seq_out])})
                   # print(len(seq_in), len(seq_out))
            return data

        train_data= create_training_data()

        # final_train_data = []
        # for example in train_data:
        #
        #     for tensor in example['input']:
        #         if tensor.numel() == 1:
        #             print(tensor.item())
        #         if len(tensor) == 0:
        #             print("empty")
        #         # else:
        #         #     print(tensor)
        #     # tensor = torch.tensor(example['input'])
        #     # print(tensor)
        #     # final_train_data.append({
        #     #     'input':torch.tensor(example['input'])
        #     # })
        #     # break;
        #

        def collate_fn(batch):
            inputs = [item['input'] for item in batch]
            targets = [item['target'] for item in batch]

            # Pad sequences to the maximum length in the batch
            padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
            padded_targets = torch.stack(targets)

            return {'input': padded_inputs, 'target': padded_targets}
        # train_data = CustomDataset(train_data)

        train_data = torch.utils.data.DataLoader(train_data,batch_size=64,collate_fn=collate_fn,shuffle=True)

        # for batch in train_data:
        #     print(batch['input'].size(), batch['target'].size())





        return {'train_data': train_data, 'word_to_one_hot':word_to_one_hot, 'one_hot_to_word':one_hot_to_word}