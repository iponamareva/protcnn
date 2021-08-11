import numpy as np
import pandas as pd
import tensorflow as tf

import json
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from tqdm import tqdm

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

from utils import dump_history, dump_histories, plot, plot_history, make_plots
from data_loaders import read_data, calc_unique_cls, read_joined_data
from preprocessing import create_dict, process_activations_dict, load_file_as_df, load_all_files_in_dir, merge_and_preprocess, integer_encoding, encode_and_pad, expand_sparse_activations_from_df
from layers import DataGenerator, residual_block, residual_block_st, app_hist
from model_utils import loss_with_params, accuracy


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Imports successful")

nclasses = 2000

print('Loading and preprocessing data')

# df_train = read_joined_data('train', 'joined')
# df_val = read_joined_data('dev', 'joined')
#print('Train size:', len(df_train)) 
#print('Val size:', len(df_val))

#classes = df_train['class_index'].value_counts()[:nclasses].index.tolist()
#print('Num classes remained:', len(classes))


#X_val = encode_and_pad(df_val)
#y_val = expand_sparse_activations_from_df(df_val, alpha=0.125)

params = {'dim': (100, 21),
          'batch_size': 256,
          'shuffle': True}

training_generator = DataGenerator("train", **params)

for batch in training_generator:
    X, y = batch
    print(X.shape, y.shape)
    break


'''

params = {'dim': (100, 21),
          'batch_size': 4,
          'shuffle': True}

training_generator = DataGenerator(mode="train", train_sm=train_sm, POS_FOR_CLASSES=POS_FOR_CLASSES, **params)
print('Len of training generator', len(training_generator))

print('Preprocessing successful')

exp_name = "exp_177"

lr = 0.001
filts = 16
print("Exp:", exp_name)

nclasses = 2000


prop = 0.0
x_student_input = Input(shape=(100, 21))
conv = Conv1D(filts, 1, padding='same')(x_student_input)
res1 = residual_block_st(conv, filts, 3)
x = MaxPooling1D(3)(res1)
x = Dropout(0.5)(x)
x = Flatten()(x)
x_student_output = Dense(nclasses, activation='softmax', kernel_regularizer=l2(0.0001))(x) 

model = Model(inputs=x_student_input, outputs=x_student_output)
model.compile(optimizer='adam', loss=loss_with_params(prop), metrics=[accuracy])

print(model.summary())
#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

history = model.fit(x=training_generator,
                    validation_data=(X_val, y_val),
                    use_multiprocessing=True,
                    workers=6,
                    epochs=2)

for batch in training_generator:
    X, y = batch
    print(X.shape, y.shape)
    break
    print('batch generated')


'''


