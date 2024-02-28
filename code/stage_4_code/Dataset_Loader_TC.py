from code.base_class.dataset import dataset
import os
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import torch.utils.data
from transformers import BertTokenizer, BertModel
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import tensorflow as tf


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
        # i = 0
        for filename in os.listdir(directory):
            # if i == 10:
            #     break
            # print(filename)
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
            porter = PorterStemmer()
            stemmed = [porter.stem(word) for word in words]
            max_len = max(max_len, len(stemmed))
            stemmed = ' '.join(stemmed)
            if sentiment == 'positive':
                data.append({'text': stemmed, 'label': torch.tensor(1)})
            else:
                data.append({'text': stemmed, 'label': torch.tensor(0)})

            # i += 1
        return data, max_len

    def batches_embeddings(self, batch):
        X, Y = [], []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        # print(X)
        for item in batch:
            text = item['text']
            label = item['label']
            # 1
            X.append(text)
            Y.append(label)
        tokens = tokenizer.batch_encode_plus(X, padding='longest', truncation=True, return_tensors='pt',
                                             add_special_tokens=True)
        inputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
        X = inputs.last_hidden_state
        Y = torch.tensor(Y)
        return {'text': X, 'label': Y}

    def embedding_text(self, data, max_len):
        return
        # df = pd.DataFrame(data)
        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(df['text'])
        #
        # word_index = tokenizer.word_index
        # vocab_size = len(word_index)
        # sequences = tokenizer.texts_to_sequences(df['text'])
        # padded_seq = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        # embedding_index = {}
        # with open('../../script/stage_4_script/.vector_cache/glove.6B.50d.txt', encoding='utf-8') as f:
        #     for line in f:
        #         values = line.split()
        #         word = values[0]
        #         coefs = np.asarray(values[1:], dtype='float32')
        #         embedding_index[word] = coefs
        # embedding_matrix = np.zeros((vocab_size + 1, 50))
        # for word, i in word_index.items():
        #     embedding_vector = embedding_index.get(word)
        #     if embedding_vector is not None:
        #         embedding_matrix[i] = embedding_vector
        # print(f'Loaded {vocab_size + 1}')
        # print(f'Embedding Matrix Shape: {embedding_matrix.shape}')
        #
        # embedded_sequences = []
        # for seq in padded_seq:
        #     embedded_seq = []
        #     for word_idx in seq:
        #         embedding_vector = embedding_matrix[word_idx]
        #         embedded_seq.append(embedding_vector)
        #     embedded_seq = np.array(embedded_seq)
        #     embedded_seq = torch.tensor(embedded_seq, dtype=torch.float32)
        #     embedded_sequences.append(embedded_seq)
        #
        # df['text'] = embedded_sequences
        # df_dict = df.to_dict(orient='records')
        # return df_dict

    def load_glove_embeddings(self, file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    def data_to_glove(self, data, embeddings, embedding_dim):
        glove_embeddings = []
        for review in data:
            embedding_matrix = np.zeros((20, embedding_dim), dtype="float32")
            text = review['text']
            words = text.split()
            for j in range(len(words)):
                if j > 19:
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
        self.max_length = max(test_max_len_pos, test_max_len_neg,train_max_len_pos, train_max_len_neg)
        glove_embeddings = self.load_glove_embeddings('../../script/stage_4_script/.vector_cache/glove.6B.50d.txt')
        train_data = self.data_to_glove(train_data, glove_embeddings, 50)
        test_data = self.data_to_glove(test_data, glove_embeddings, 50)
        # train_data = self.embedding_text(train_data, self.max_length)
        # test_data = self.embedding_text(test_data, self.max_length)
        # print(train_data)
        # print(test_data)
        # GET PLAIN TEXT FROM THEM
        # train_text = self.get_plain_text(train_data)
        # test_text = self.get_plain_text(test_data)
        #
        # # GET EMBEDDINGS
        # train_inputs = self.get_embeddings(train_text)
        # test_inputs = self.get_embeddings(test_text)

        # ADD THEM BACK INTO train and test data
        # train_data = self.change_dataset(train_inputs, train_data)
        # test_data = self.change_dataset(test_inputs, test_data)
        print(train_data)

        train_data = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
        # for batch in train_data:
        #     texts = batch['text']
        #     labels = batch['label']

        return {'train_data': train_data, 'test_data': test_data}