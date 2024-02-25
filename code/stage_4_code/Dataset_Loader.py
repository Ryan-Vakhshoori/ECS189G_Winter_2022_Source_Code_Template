from code.base_class.dataset import dataset
import os
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import torch.utils.data

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def clean(self, directory, sentiment):
        data = []
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            file = open(f, 'rt')
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
            if sentiment == 'positive':
                data.append({'words': stemmed, 'label': 1})
            else:
                data.append({'words': stemmed, 'label': 0})

        return data

    def load(self):
        print('loading data...')
        train_neg_directory = self.dataset_source_folder_path + '/train/neg'
        train_pos_directory = self.dataset_source_folder_path + '/train/pos'
        test_neg_directory = self.dataset_source_folder_path + '/test/neg'
        test_pos_directory = self.dataset_source_folder_path + '/test/pos'
        data = {"train": [self.clean(train_neg_directory, "negative") +
                          self.clean(train_pos_directory, "positive")],
                "test": [self.clean(test_neg_directory, "negative") +
                         self.clean(test_pos_directory, "positive")]}
        train_data = torch.utils.data.DataLoader(data['train'], batch_size=64, shuffle=True)
        test_data = torch.utils.data.DataLoader(data['test'], batch_size=64, shuffle=False)
        return {'train_data': train_data, 'test_data': test_data}