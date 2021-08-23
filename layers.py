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
from data_loaders import *


def residual_block(data, filters, d_rate):
  shortcut = data
  bn1 = BatchNormalization()(data)
  act1 = Activation('relu')(bn1)
  conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)
  bn2 = BatchNormalization()(conv1)
  act2 = Activation('relu')(bn2)
  conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)
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

def make_model(args):
  x_student_input = Input(shape=(100, 21))
  conv = Conv1D(args.num_filters, 1, padding='same')(x_student_input)
  # res1 = residual_block_st(conv, args.num_filters, 3)
  res1 = residual_block(conv, args.num_filters, 2)
  res2 = residual_block(res1, args.num_filters, 3)
  x = MaxPooling1D(3)(res2)
  x = Dropout(0.5)(x)
  x = Flatten()(x)
  x_student_output = Dense(NCL, activation='softmax', kernel_regularizer=l2(args.learning_rate))(x)

  model = Model(inputs=x_student_input, outputs=x_student_output)

  return model


