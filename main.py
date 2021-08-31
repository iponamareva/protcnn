import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

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
from tensorflow.keras.callbacks import CSVLogger

from utils import dump_history, dump_histories, plot, plot_history, make_plots, make_training_log, app_hist
from data_loaders import calc_unique_cls
from preprocessing import preprocess_dfs,  make_dataset
from layers import residual_block, residual_block_st, make_model
from model_utils import loss_with_params, accuracy
from constants import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Imports successful")

parser = argparse.ArgumentParser(description="Arguments for run")
parser.add_argument("-na", "--exp-name", type=str)
parser.add_argument("-ep", "--epochs", nargs="?", type=int, default=10)
parser.add_argument("-tp", "--true-prop", nargs="?", type=float, default=0.0)
parser.add_argument("-lr", "--learning-rate", nargs="?", type=float, default=0.0001)
parser.add_argument("-fs", "--num-filters", nargs="?", type=int, default=16)
parser.add_argument("-al", "--alpha", nargs="?", type=float, default=1.0)

args = parser.parse_args()

print('Loading and preprocessing data')

X_train, y_train = preprocess_dfs("train", alpha=args.alpha)
X_val, y_val = preprocess_dfs("dev", alpha=args.alpha)

train_dataset = make_dataset(X_train, y_train)
val_dataset = make_dataset(X_val, y_val)

print("Experiment name:", args.exp_name)

model = make_model(args)
model.compile(optimizer='adam', loss=loss_with_params(args.true_prop), metrics=[accuracy])

args_dict = {"exp_name" : args.exp_name,
        "epochs" : args.epochs,
        "lr" : args.learning_rate,
        "true p" : args.true_prop,
        "filters" : args.num_filters,
        "alpha" : args.alpha}

log_file_name = "training_logs/"+args.exp_name+".log"

make_training_log(args_dict, log_file_name)
with open(log_file_name, "a") as f:
    print(model.summary(), file=f)

filename= args.exp_name + '.csv'
history_logger = CSVLogger(filename, separator=",", append=True)

history = model.fit(x=train_dataset,
                    validation_data=val_dataset,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks=[history_logger],
                    epochs=args.epochs)

dump_history(history, "histories/"+args.exp_name+".pkl")




