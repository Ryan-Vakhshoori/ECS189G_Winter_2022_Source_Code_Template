from code.base_class.dataset import dataset
import os
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import torch.utils.data
from transformers import BertTokenizer, BertModel

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    max_length = None
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def clean(self, directory, sentiment):
        data = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        max_len = 0
        for filename in os.listdir(directory):
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
            stemmed = ' '.join(stemmed)
            max_len = max(max_len, len(stemmed))
            # tokens = tokenizer.batch_encode_plus(stemmed, padding=True, truncation=True, return_tensors='pt',
            #                                      add_special_tokens=True)
            # inputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
            # final_input = inputs.last_hidden_state
            # print(final_input.shape)
            # print(stemmed)
            if sentiment == 'positive':
                data.append({'text': stemmed, 'label': torch.tensor(1)})
            else:
                data.append({'text': stemmed, 'label': torch.tensor(0)})
            break
        return data, max_len

    # def get_plain_text(self, data):
    #     text_keeper = []
    #     for data in data:
    #         text_keeper.append(data['text'])
    #     return text_keeper

    # def get_embeddings(self, text_keeper):
    #   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #         model = BertModel.from_pretrained('bert-base-uncased')
    #       tokens = tokenizer.batch_encode_plus(stemmed, padding=True, truncation=True, return_tensors='pt',
    #             #                                      add_special_tokens=True)
    #             # inputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
    #     # print(f'Inputs Shape: {inputs.last_hidden_state.shape}')
    #     # print(f'Vocab Size: {tokenizer.vocab_size}')
    #     # print(f'Get Vocab Vectors:{list(tokenizer.vocab.keys())[5000:5020]}')
    #     return inputs

    def change_dataset(self, inputs, data):
        for k, v in zip(inputs.last_hidden_state, data):
            v['item'] = k
            # print(v['text'].shape)
        # print(f'Data: {data}')
        return data

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
        # print(Y)
        tokens = tokenizer.batch_encode_plus(X, padding=True, truncation=True, return_tensors='pt',
                                             add_special_tokens=True)
        inputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
        X = inputs.last_hidden_state
        Y = torch.tensor(Y)
        # print(X)
        # print(Y)
        return {'text': X, 'label': Y}
        # print(batch)
        # # Y, X = list(zip(*batch))
        # X, Y = [item['text'] for item in batch], [item['label'] for item in batch]
        # # print(X)
        # # print(Y)
        # # text = *batch['text']
        # # labels = *batch['labels']
        # # print(text)
        # # print(labels)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained('bert-base-uncased')
        # tokens = tokenizer.batch_encode_plus(X, padding=True, truncation=True, return_tensors='pt',
        #                                      add_special_tokens=True)
        # inputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
        # print(f'Inputs Shape: {inputs.last_hidden_state.shape}')
        # X = self.change_dataset(inputs, batch)
        # print(X)
        # print(Y)
        # return X, Y
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
        # print(train_data)

        train_data = torch.utils.data.DataLoader(train_data, batch_size=64, collate_fn=self.batches_embeddings, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_data, batch_size=64, collate_fn=self.batches_embeddings, shuffle=False)
        # for batch in train_data:
        #     texts = batch['text']
        #     print(batch['text'].shape)
        #     labels = batch['label']
        #     print(batch['label'].shape)
        return {'train_data': train_data, 'test_data': test_data}