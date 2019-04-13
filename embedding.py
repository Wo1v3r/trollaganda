"""
This module contains the text pre-process, and embed it.
"""
import os
import zipfile
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


class PreProcess(object):
    """
    This class will pre-process pandas data frame
    """
    def __init__(self, data, textfield):
        self.data = data
        self.textfield = textfield

    def process_text(self):
        self.data[self.textfield] = self.data[self.textfield].str.replace(r"http\S+", "LINK")
        self.data[self.textfield] = self.data[self.textfield].str.replace(r"@\S+", "TAG")
        self.data[self.textfield] = self.data[self.textfield].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
        self.data[self.textfield] = self.data[self.textfield].str.replace(r"@", "AT")
        self.data[self.textfield] = self.data[self.textfield].str.lower()
        return self.data

    def see_data_head(self):
        self.data.head()


class ReadFile(object):
    """
    Reading the datasets class.
    The data set should be two columns,
    tweet and classification(1 is troll)
    """
    def __init__(self, path, split=None):
        self.path = path
        self.split = split
        self.data = None

    def readfile(self):
        self.data = pd.read_csv(self.path, delimiter=",", encoding="utf8", names=["message", "isTroll"])
        self.data.message = self.data.message.astype(str)

        self.data = self.data.iloc[:self.split] if self.split > 0 else self.data

    def distribution_plot(self):
        if self.data is not None:
            sns.countplot(self.data.isTroll)
            plt.xlabel("Label")
            plt.title("Number of troll or not messages")


class PrepareEmbedding(object):
    """
    This class is used to create the embedding on the data
    """
    def __init__(self, X, Y, embedded_path, test_size=0.15):
        self.X = X
        self.Y = Y
        self.test_size = test_size
        self.embedded_path = embedded_path
        self.pre_train = None
        self._prepare_labels()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=self.test_size)
        self.train_embedding_weights = None
        self.train_cnn_data = None
        self.test_cnn_data = None
        self.train_word_index = None

    def _prepare_labels(self):
        le = LabelEncoder()
        self.Y = le.fit_transform(self.Y)
        self.Y = self.Y.reshape(-1, 1)

    def _tokenize_messages(self):
        # TOKENIZING THE TEXT
        regextokenizer = RegexpTokenizer(r"\w+")
        self.X_train = self.X_train.apply(regextokenizer.tokenize)
        # delete Stop Words
        self.X_train = self.X_train.apply(lambda vec: [word for word in vec if word not in config.stopwords0])
        return self.X_train

    def print_info(self):
        train_tokens = self._tokenize_messages()
        all_training_words = [word for tokens in train_tokens for word in tokens]
        training_sentence_lengths = [len(tokens) for tokens in train_tokens]
        training_vocab = sorted(list(set(all_training_words)))

        print("Total: %s words, vocabulary size of %s" % (len(all_training_words), len(training_vocab)))
        print("Max sentence length is %s" % max(training_sentence_lengths))

    def load_word_2_vec(self):
        print("Loading W2V")
        self.pre_train = gensim.models.KeyedVectors.load_word2vec_format(self.embedded_path, binary=True)
        print("W2V Loaded")

    def load_glove(self):
        print("Loading GloVe")
        self.pre_train = {}
        # https: // nlp.stanford.edu / projects / glove /
        if not os.path.isfile(self.embedded_path):  # should be .txt
            zip_ref = zipfile.ZipFile(self.embedded_path, 'r')  # should be .zip
            zip_ref.extractall('./')
            zip_ref.close()
        f = open(self.embedded_path)
        for line in f:
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            self.pre_train[word] = coefs
        f.close()
        print("GloVe data loaded")

    def train(self):
        try:
            if not self.pre_train:
                raise Exception("Pre trained vocabulary isn't loaded.")
        except Exception as e:
            print(e)
            return

        tokenizer = Tokenizer(num_words=config.MAXVOCABSIZE, lower=True, char_level=False)
        tokenizer.fit_on_texts(self.X_train)
        training_sequences = tokenizer.texts_to_sequences(self.X_train)

        self.train_word_index = tokenizer.word_index
        self.train_cnn_data = pad_sequences(training_sequences, maxlen=config.MAXSEQLENGTH)

        self.train_embedding_weights = np.zeros((len(self.train_word_index) + 1, config.EMBEDDINGDIM))
        for word, index in self.train_word_index.items():
            self.train_embedding_weights[index, :] = self.pre_train[word] if word in self.pre_train \
                else np.random.rand(config.EMBEDDINGDIM)

        # Prepare the test data
        test_sequences = tokenizer.texts_to_sequences(self.X_test)
        self.test_cnn_data = pad_sequences(test_sequences, maxlen=config.MAXSEQLENGTH)

        print("Found {} unique tokens.".format(len(self.train_word_index)))

    def release_pre_trained(self):
        del self.pre_train
        self.pre_train = None
