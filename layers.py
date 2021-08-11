import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Embedding, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import activations

from preprocessing import *
from model_utils import *
from constants import *

class DataGenerator(Sequence):
    def __init__(self, mode, batch_size=256, dim=(100, 21), shuffle=True):
        
        print('Initializing data generator in mode', mode)
        if mode=='train':
            kaggle_df = read_data(kaggle_data_path, mode)
            google_df = load_all_files_in_dir(google_data_path+mode)
            self.df = merge_and_preprocess(kaggle_df, google_df)
            print(len(self.df))
        else:
            print("Error: unknown mode")
        self.df = df

        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.df)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Select and process data
        df_batch = self.df.loc[indexes]

        X = encode_and_pad(df_batch)
        y = expand_sparse_activations_from_df(df_batch, alpha=1.0)

        return (X, y)


def residual_block(data, filters, d_rate):
  shortcut = data
  bn1 = BatchNormalization()(data)
  act1 = Activation('relu')(bn1)
  conv1 = Conv1D(filters, 5, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)
  bn2 = BatchNormalization()(conv1)
  act2 = Activation('relu')(bn2)
  conv2 = Conv1D(filters, 5, padding='same', kernel_regularizer=l2(0.001))(act2)
  x = Add()([conv2, shortcut])
  return x


def residual_block_st(data, filters, d_rate):
  shortcut = data
  bn1 = BatchNormalization()(data)
  act1 = Activation('relu')(bn1)
  conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)
  bn2 = BatchNormalization()(conv1)
  act2 = Activation('relu')(bn2)
  conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)
  x = Add()([conv2, shortcut])
  return x


def app_hist(f, losses, accs, val_losses, val_accs, th, th2):
    with open(f, 'rb') as file:
        hist = pickle.load(file)
    losses.append(hist['loss'][th:th2])
    accs.append(hist['accuracy'][th:th2])
    val_losses.append(hist['val_loss'][th:th2])
    val_accs.append(hist['val_accuracy'][th:th2])
    return losses, accs, val_losses, val_accs



