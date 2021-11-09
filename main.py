print('Starting', flush=True)

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import time

import json
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from tqdm import tqdm

from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow import GradientTape
from tensorflow.keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

from utils import dump_history, dump_histories, plot, plot_history, make_plots, make_training_info, app_hist
from model_utils import loss_with_params, accuracy, AccCustomMetric
from preprocessing import preprocess_dfs,  make_dataset, preprocess_dfs_only_true, make_dataset_only_true
from constants import *
from layers import ProtCNNModel

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), flush=True)

parser = argparse.ArgumentParser(description="Arguments for run")
parser.add_argument("-na", "--exp-name", nargs="?", type=str, default="exp")
parser.add_argument("-ep", "--epochs", nargs="?", type=int, default=10)
parser.add_argument("-tp", "--true-prop", nargs="?", type=float, default=0.0)
parser.add_argument("-lr", "--learning-rate", nargs="?", type=float, default=0.0001)
parser.add_argument("-fs", "--num-filters", nargs="?", type=int, default=16)
parser.add_argument("-al", "--alpha", nargs="?", type=float, default=1.0)
parser.add_argument("-ml", "--max-length", nargs="?", type=int, default=100)
parser.add_argument("-ks", "--kernel-size", nargs="?", type=int, default=5)
parser.add_argument("-bs", "--batch-size", nargs="?", type=int, default=256)
parser.add_argument("-fl", "--flag", nargs="?", type=char, default="D")

args = parser.parse_args()

print('Loading and preprocessing data', flush=True)
print("Experiment name:", args.exp_name)

if args.flag == 'D':
  X_train, X_train_lengths, y_train = preprocess_dfs("train", max_length=args.max_length, alpha=args.alpha)
  X_val, X_val_lengths, y_val = preprocess_dfs("dev", max_length=args.max_length, alpha=args.alpha)
  train_dataset, TRAIN_SIZE = make_dataset(X_train, X_train_lengths, y_train, args=args)
  val_dataset, VAL_SIZE = make_dataset(X_val, X_val_lengths, y_val, args=args)

elif args.flag = 'C':
  X_train, X_train_lengths, y_train = preprocess_dfs_only_true("train", max_length=args.max_length, alpha=args.alpha)
  X_val, X_val_lengths, y_val = preprocess_dfs_only_true("dev", max_length=args.max_length, alpha=args.alpha)

  train_dataset, TRAIN_SIZE = make_dataset_only_true(X_train, X_train_lengths, y_train, args=args)
  val_dataset, VAL_SIZE = make_dataset_only_true(X_val, X_val_lengths, y_val, args=args)

print("TRAIN_SIZE:", TRAIN_SIZE, "VAL_SIZE:", VAL_SIZE)

make_training_info(args)
filename = args.exp_name + ".log"
filepath = os.path.join(LOG_DIR, filename)

filename = CSV_LOGS_DIR + '/' + args.exp_name + '.csv'
history_logger = CSVLogger(filename, separator=",", append=True)


''' MAKE AND COMPILE MODEL '''
model = ProtCNNModel(args)

for batch in val_dataset.take(1):
  X, X_l, y = batch
  preds = model((tf.cast(X, tf.float32), X_l)) 
  print('COUNT PARAMS', model.count_params())
  
optimizer = Adam(learning_rate=args.learning_rate)
if args.flag == 'D':
  ce_loss_fn = loss_with_params(args.true_prop)
  train_acc_metric = AccCustomMetric()
  val_acc_metric = AccCustomMetric()
elif args.flag == 'C':
  ce_loss_fn = SparseCategoricalCrossentropy()
  train_acc_metric = Accuracy()
  val_acc_metric = Accuracy()
 
loss_metric = Mean()

for epoch in range(args.epochs):
  print("Start of epoch %d" % (epoch,))
  start_time = time.time()

  for step, batch_train in enumerate(train_dataset):
    X, X_lengths, y = batch_train

    with GradientTape() as tape:
      preds = model((tf.cast(X, tf.float32), X_lengths), training=True)
      loss = ce_loss_fn(y, preds)
    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # update metric
    train_acc_metric.update_state(y, preds)
    loss_metric(loss)

    if step % 100 == 0:
      print("step %d: mean loss = %.4f" % (step, loss_metric.result()), flush=True)
       
  train_acc = train_acc_metric.result()
  train_acc_metric.reset_states()

  for step, batch_val in enumerate(val_dataset):
    X_val, X_lengths_val, y_val = batch_val
    preds_val = model((tf.cast(X_val, tf.float32), X_lengths_val), training=False)
    val_acc_metric.update_state(y_val, preds_val)
  
  val_acc = val_acc_metric.result()
  val_acc_metric.reset_states()

  print("Training acc over epoch: %.4f" % (float(train_acc) / TRAIN_SIZE,), "Validation acc: %.4f" % (float(val_acc) / VAL_SIZE,), "Time taken for epoch: %.2fs" % (time.time() - start_time))


