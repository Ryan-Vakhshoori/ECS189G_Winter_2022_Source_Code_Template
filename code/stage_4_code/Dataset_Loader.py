from code.base_class.dataset import dataset
import os
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        neg_directory = self.dataset_source_folder_path + '/neg'
        for filename in os.listdir(neg_directory):
            f = os.path.join(neg_directory, filename)
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
            print(stemmed[:100])
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
        # X = []
        # y = []
        # f = open(self.dataset_source_folder_path + '/neg, 'r')
        # for line in f:
        #     line = line.strip('\n')
        #     elements = [int(i) for i in line.split(' ')]
        #     X.append(elements[:-1])
        #     y.append(elements[-1])
        # f.close()
        # return {'X': X, 'y': y}