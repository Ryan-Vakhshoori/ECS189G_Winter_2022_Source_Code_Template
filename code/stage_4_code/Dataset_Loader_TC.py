from code.base_class.dataset import dataset
import os
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import torch.utils.data
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    max_length = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def clean(self, directory, sentiment):
        data = []
        max_len = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            file = open(f, 'rt', encoding='utf-8')
            text = file.read()
            file.close()
            # split into words by white space
            tokens = word_tokenize(text)
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            # filter out stop words
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            max_len = max(max_len, len(words))
            words = ' '.join(words)
            if sentiment == 'positive':
                data.append({'text': words, 'label': torch.tensor(1)})
            else:
                data.append({'text': words, 'label': torch.tensor(0)})
        return data, max_len


    def load_glove_embeddings(self, file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings


    def data_to_glove(self, data, embeddings, embedding_dim, max_len):
        glove_embeddings = []
        for review in data:
            embedding_matrix = np.zeros((max_len, embedding_dim), dtype="float32")
            text = review['text']
            words = text.split()
            for j in range(len(words)):
                if j == max_len:
                    break
                if words[j] in embeddings:
                    embedding_matrix[j] = embeddings[words[j]]
            glove_embeddings.append({'embedding': embedding_matrix, 'label': review['label']})
        return glove_embeddings


    def load(self):
        print('loading data...')
        train_neg_directory = self.dataset_source_folder_path + '/train/neg'
        train_pos_directory = self.dataset_source_folder_path + '/train/pos'
        test_neg_directory = self.dataset_source_folder_path + '/test/neg'
        test_pos_directory = self.dataset_source_folder_path + '/test/pos'
        train_data_neg, train_max_len_neg = self.clean(train_neg_directory, "negative")
        train_data_pos, train_max_len_pos = self.clean(train_pos_directory, "positive")
        train_data = train_data_pos + train_data_neg
        test_data_neg, test_max_len_neg = self.clean(test_neg_directory, "negative")
        test_data_pos, test_max_len_pos = self.clean(test_pos_directory, "positive")
        test_data = test_data_pos + test_data_neg
        self.max_length = max(test_max_len_pos, test_max_len_neg, train_max_len_pos, train_max_len_neg)
        glove_embeddings = self.load_glove_embeddings('../../script/stage_4_script/.vector_cache/glove.6B.100d.txt')
        train_data = self.data_to_glove(train_data, glove_embeddings, 100, 130)
        test_data = self.data_to_glove(test_data, glove_embeddings, 100, 130)

        train_data = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

        return {'train_data': train_data, 'test_data': test_data}