import json
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import pickle

import tensorflow as tf
from tensorflow.sparse import SparseTensor
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from utils import renorm
from constants import *

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index + 1

  return char_dict

char_dict = create_dict(codes)

def densify(X, y):
    X = tf.one_hot(X, 21, on_value=1, off_value=0)
    return X, tf.sparse.to_dense(y)

def make_dataset(X, y, batch_size=256):
    dataset = Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(256)
    dataset = dataset.map(densify)
    return dataset


def process_activations_dict():
    d = {}
    filename = google_data_path + 'blundell_pfam_32_vocab.json'
    with open(filename, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        for i, elem in enumerate(data):
            d[elem] = i

    return d, data

def read_data(partition, data_path = 'random_split/'):
  data = []
  for fn in os.listdir(os.path.join(data_path, partition)):
    with open(os.path.join(data_path, partition, fn)) as f:
      data.append(pd.read_csv(f, index_col=None))
  return pd.concat(data)

def read_joined_data(partition, data_path):
  with open(os.path.join(data_path, partition+'.csv')) as f:
    data = pd.read_csv(f, index_col=None)
    return data

def load_file_as_df(fname):
    ''' loading single file as a pd dataframe '''


    seq_names, seqs, acts = [], [], []
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            seqs.append(data['sequence'])
            seq_names.append(data['sequence_name'])
            acts.append(data['activations'])

    res = {'sequence_name' : seq_names,
           'sequence' : seqs,
           'activations' : acts}

    df = pd.DataFrame(res, columns = ['sequence_name', 'sequence', 'activations'])

    return df

def load_all_files_in_dir(p, n_files=-1) :
    df_list = []
    files = [f for f in listdir(p) if isfile(join(p, f))]
    print('Dir:', p, 'contains', len(files), 'files.')
    if n_files != -1:
        files = files[:n_files]
        print("Remained:", n_files, "files")
    
    for f in files:
        filename = join(p, f)
        df = load_file_as_df(filename)
        df_list.append(df)

    df_conc = pd.concat(df_list)
    print("Files loaded and concatenated")
    return df_conc


def preprocess_dfs(mode="train", max_length=100, alpha=1.0):
    TRUNCATED_DATA = (NCL < NCL_MAX)

    POS_FOR_CLASSES, classes_list = process_activations_dict()
    UPD_POS_FOR_CLASSES = {}

    kaggle_df = read_data(mode, kaggle_data_path)
    google_df = load_all_files_in_dir(google_data_path+mode)

    df_res = kaggle_df.merge(google_df, how='left', 
                                        left_on='sequence_name', 
                                        right_on='sequence_name',
                                        suffixes=('','_conc'))
    df_res.drop(columns=['sequence_conc', 'aligned_sequence', 'family_id'], inplace=True)
    df_res.dropna(inplace=True)
    df_res = df_res.reset_index()

    if TRUNCATED_DATA:
        print("Truncation: top", NCL, "classes")
        with open("classes.pkl", "rb") as f:
            classes = pickle.load(f)
        classes = classes[:NCL]
        df_res = df_res.loc[df_res['family_accession'].isin(classes)].reset_index()

        for i, class_name in enumerate(classes):
            class_name = class_name.split('.')[0]
            cur_pos = POS_FOR_CLASSES[class_name]
            UPD_POS_FOR_CLASSES[cur_pos] = i
        
    

    def integer_encoding(df):
        encode_list = []
        for row in df['sequence'].values: # 'sequence'
            row_encode = []
            for code in row:
                row_encode.append(char_dict.get(code, 0))
            encode_list.append(np.array(row_encode))
  
        return encode_list

    def encode_and_pad(df):
        x_encoded = integer_encoding(df)
        x_padded = pad_sequences(x_encoded, maxlen=max_length, padding='post', truncating='post')

        return x_padded
    
    def get_indices_and_values(df):
        indices, values = [], []
        shape = [len(df), 2 * NCL]
        print("Making SparseTensor of shape:", shape)

        for i, row in df.iterrows():

            class_name = row['family_accession'].split('.')[0]
            class_index = POS_FOR_CLASSES[class_name]
            if TRUNCATED_DATA:
                class_index = UPD_POS_FOR_CLASSES[class_index]

            indices.append([i, class_index])
            values.append(1.0)
            
            acts = row['activations']
            new_values = []
            new_indices = []
 
            for elem in acts:
                if (elem[0] < NCL_MAX): # Check that it's real class
                    act_pos = elem[0]

                    if TRUNCATED_DATA: # Update class index if needed
                        if act_pos in UPD_POS_FOR_CLASSES: # If this class is present in updated dataset
                            act_pos = UPD_POS_FOR_CLASSES[act_pos]
                    
                            new_indices.append([i, NCL + act_pos])
                            new_values.append(elem[1])
                    else:
                        indices.append([i, NCL + act_pos])
                        new_values.append(elem[1])
            
            if TRUNCATED_DATA:
                new_indices = sorted(new_indices, key=lambda x: x[1])
                indices.extend(new_indices)

            new_values = renorm(new_values, alpha=alpha)
            values.extend(new_values)

        print(len(indices), len(values), shape)
        return indices, values, shape

    X = encode_and_pad(df_res)
    indices, values, shape = get_indices_and_values(df_res)
    y = SparseTensor(indices, values, shape)

    return X, y
