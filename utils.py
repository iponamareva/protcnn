import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from constants import *

def app_hist(f, losses, accs, val_losses, val_accs, th, th2):
    with open(f, 'rb') as file:
        hist = pickle.load(file)
    losses.append(hist['loss'][th:th2])
    accs.append(hist['accuracy'][th:th2])
    val_losses.append(hist['val_loss'][th:th2])
    val_accs.append(hist['val_accuracy'][th:th2])
    return losses, accs, val_losses, val_accs


def make_training_log(args):
  filename = args.exp_name + ".log"
  filepath = os.path.join(LOG_DIR, filename)
  d = vars(args)

  with open(filepath, "w") as f:
    for key in d:
        print(key, "\t", d[key], file=f)

    print("NCL\t", NCL)

def renorm(v, alpha=1):
    p = np.power(v, [alpha])
    return p / p.sum()

# for 1 model
def dump_history(history, path):
    loss = history.history["loss"]
    accuracy = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_accuracy = history.history["val_accuracy"]

    with open(path, 'wb') as pickle_file:
        pickle.dump(
            {"loss" : loss, "accuracy": accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy},
            pickle_file)
        
# for training multiple models
def dump_histories(histories, path):
  dump_data = []
  for history in histories:
      loss = history.history["loss"]
      accuracy = history.history["accuracy"]
      val_loss = history.history["val_loss"]
      val_accuracy = history.history["val_accuracy"]

      
      dump_data.append({"loss" : loss, "accuracy": accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})
    
  with open(path, 'wb') as pickle_file:
      pickle.dump(dump_data, pickle_file)

def plot(vars, labels, color, title):
    # plt.figure(figsize=(6, 4))
    plt.grid()
    for i, var in enumerate(vars):
        plt.plot(var, '-x', label=labels[i], color=color[i])
    plt.title(title)
    plt.legend()

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def make_plots(th=0, c=None, step=50, value=0.9426):    
    plt.style.use('seaborn-white')
    colors = ['darkblue', 'mediumblue',  'seagreen', 'green', 'greenyellow', 'yellow']
    if c is not None:
      colors=c
    f, axs = plt.subplots(2,2,figsize=(20,20))
    plt.subplot(2, 2, 1)
    plot(losses, names, color=colors, title='Losses on train')
    labels = [th + x for x in range(0, len(val_accs[0]), 5)]
    plt.xticks(np.arange(0, len(val_accs[0]), step=step), labels)

    plt.subplot(2, 2, 2)
    plot(accs, names, color=colors, title='Accuracies on train')
    labels = [th + x for x in range(0, len(val_accs[0]), 5)]
    plt.xticks(np.arange(0, len(val_accs[0]), step=step), labels)

    plt.subplot(2, 2, 3)
    plot(val_losses, names, color=colors, title='Losses on validation')
    labels = [th + x for x in range(0, len(val_accs[0]), 5)]
    plt.xticks(np.arange(0, len(val_accs[0]), step=step), labels)

    plt.subplot(2, 2, 4)
    plot(val_accs, names, color=colors, title='Accuracies on validation')
    labels = [th + x for x in range(0, len(val_accs[0]), 5)]
    plt.xticks(np.arange(0, len(val_accs[0]), step=step), labels)
