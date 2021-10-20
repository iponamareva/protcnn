import numpy as np
import pandas as pd
import tensorflow as tf
import math

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
from tensorflow.keras.layers import Embedding, Masking, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.backend import int_shape
from tensorflow import tile as tile_fn
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

from preprocessing import *
from model_utils import *
from constants import *
from data_loaders import *

def _tile(x, embedding_size):
  out = tf.tile(x, [1, 1, embedding_size])
  return out


def _set_padding_to_sentinel(sequences, sequence_lengths):
  sentinel = tf.constant(0.)
  seq_dim, emb_dim = 1, 2

  longest_sequence_length = tf.shape(sequences)[seq_dim]
  emb_size = tf.shape(sequences)[emb_dim]

  seq_mask = tf.sequence_mask(sequence_lengths, longest_sequence_length)
  seq_mask = tf.expand_dims(seq_mask, [emb_dim])
  is_not_padding = tf.tile(seq_mask, [1, 1, emb_size])

  full_sentinel = tf.zeros_like(sequences)
  full_sentinel = full_sentinel + tf.convert_to_tensor(sentinel)

  per_location_representations = tf.where(is_not_padding, sequences, full_sentinel)

  return per_location_representations


class Conv1DCustom(Layer):
  def __init__(self, filters, kernel_size, dilation_rate):
    super(Conv1DCustom, self).__init__()
    self.filters = filters
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate

  def build(self, input_shape):
    self.conv1D = Conv1D(filters=self.filters, kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, padding='same')

  def call(self, X_packed):
    sequences, sequence_lengths = X_packed
    
    padding_zeroed = _set_padding_to_sentinel(sequences, sequence_lengths)
    conved = self.conv1D(padding_zeroed)
    re_zeroed = _set_padding_to_sentinel(conved, sequence_lengths)
    
    return re_zeroed


class ProtResBlock(Layer):
  def __init__(self, args, layer_index):
    super(ProtResBlock, self).__init__()
    
    self.layer_index = layer_index
    dilation_rate = DIL_RATES[layer_index]
    num_filters_bottleneck = math.floor(BTLNCK_F * args.num_filters)

    self.BN1 = BatchNormalization()
    self.BN2 = BatchNormalization()
    self.Conv1 = Conv1DCustom(filters=num_filters_bottleneck, kernel_size=args.kernel_size, dilation_rate=dilation_rate)
    self.Conv2 = Conv1DCustom(filters=args.num_filters, kernel_size=1, dilation_rate=1)

  # def build(self, input_shape):
  
 
  def call(self, inputs, training=None):
    sequence_features, sequence_lengths = inputs
    x = self.BN1(sequence_features, training=training)
    x = Activation('relu')(x)
    x = self.Conv1((x, sequence_lengths))
    x = self.BN2(x, training=training)
    x = Activation('relu')(x)

    x = self.Conv2((x, sequence_lengths))
    result = sequence_features + x
    
    return result
    

def _make_per_sequence_features(sequences, sequence_lengths, args):
  per_location_representations = _set_padding_to_sentinel(sequences, sequence_lengths)
  denominator = tf.cast(tf.expand_dims(sequence_lengths, axis=-1), tf.float32)**DENOM_POWER #~~~
  pooled_representation = tf.reduce_sum(per_location_representations, axis=1) / denominator
  pooled_representation = tf.identity(pooled_representation, name='pooled_representation')

  return pooled_representation


class ProtCNNModel(Model):
  def __init__(self, args, name="protcnnmodel"):
    super(ProtCNNModel, self).__init__(name=name)
    self.ConvInitial = Conv1DCustom(filters=args.num_filters, kernel_size=args.kernel_size, dilation_rate=1)
    self.ResBlock1 = ProtResBlock(args, layer_index=0)
    self.ResBlock2 = ProtResBlock(args, layer_index=1)
    self.Dense = Dense(NCL, activation='softmax')
    self.args = args
 
  def call(self, X_packed, training=None):
    X, X_lengths = X_packed

    conved = self.ConvInitial((X, X_lengths))
    res = self.ResBlock1((conved, X_lengths), training=training)
    res = self.ResBlock2((res, X_lengths), training=training)

    per_sequence_features = _make_per_sequence_features(res, X_lengths, self.args)
    res = self.Dense(per_sequence_features)

    return res

  def model(self):
    x = Input(shape=(self.args.batch_size, self.args.max_length, 21))
    return Model(inputs=[x], outputs=self.call(x))

  def build_graph(self, raw_shape):
    x = Input(shape=raw_shape)
    return Model(inputs=[x], outputs=self.call(x))

