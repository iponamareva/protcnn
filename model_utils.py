import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
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

from constants import *

def loss_with_params(true_ls_prop=1.0):
  def loss(yTrue, yPred):
        # if yTrue.shape[1] > nclasses:
            # print('!!!removing first nclasses elems')
      yPredictions = yTrue[:, nclasses:]
      yTrue = yTrue[:, :nclasses]
      
      loss2 = tf.keras.losses.categorical_crossentropy(yPredictions, yPred)
      loss1 = tf.keras.losses.categorical_crossentropy(yTrue, yPred)

      return true_ls_prop * loss1  + (1-true_ls_prop)* loss2

  return loss


def accuracy(y_true, y_pred):
    # if y_true.shape[1] > nclasses:
    y_true = y_true[:, :nclasses]
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)
