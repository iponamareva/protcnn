import json
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from utils import renorm

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index + 1

  return char_dict

char_dict = create_dict(codes)


def process_activations_dict():
    d = {}
    filename = 'blundell_seed_ensemble_activations/blundell_pfam_32_vocab.json'
    with open(filename, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        for i, elem in enumerate(data):
            d[elem] = i

    return d, data

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

def merge_and_preprocess(data_sample, df_conc, pos_dict=None):
    print("Merging and preprocessing...")
    POS_FOR_CLASSES, classes_list = process_activations_dict()
    print("Number of classes in dictionary:", len(POS_FOR_CLASSES))

    df_res = data_sample.merge(df_conc, how='left', 
                                        left_on='sequence_name', 
                                        right_on='sequence_name',
                                        suffixes=('','_conc'))
    df_res.drop(columns=['sequence_conc', 'aligned_sequence', 'family_id'], inplace=True)
    df_res.dropna(inplace=True)

    def label_class (row):
        ''' Get class labels from the dictionary. '''

        cls = row['family_accession'].split('.')[0]
        position = POS_FOR_CLASSES[cls]
        return position

    def filter_activations(row):
        ''' Filtering activations.
            This is needed because not all classes are present. '''

        acts = row['activations']
        new_acts = []
        # if this class is present in the dataset
        # (classes 17929 - 18929 are filtered out)

        for elem in acts:
            if (elem[0] < len(POS_FOR_CLASSES)):
                new_acts.append([elem[0], elem[1]])
        return new_acts

    # Labelling classes according to the initial dictionary
    df_res['class_index'] = df_res.apply(lambda row: label_class(row), axis=1)
    df_res['new_activations'] = df_res.apply(lambda row: filter_activations(row), axis=1)
    print("Done!")

    return df_res

def integer_encoding(df):
  ''' encoding all sequences in a df '''
  encode_list = []
  for row in df['sequence'].values: # 'sequence'
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list

def encode_and_pad(df):
    x_encoded = integer_encoding(df)
    max_length = 100
    x_padded = pad_sequences(x_encoded, maxlen=max_length, padding='post', truncating='post')
    x_ohe = to_categorical(x_padded)

    return x_ohe

def expand_sparse_activations_from_df(df, n_classes=2000, alpha=0.125):
    ''' expanding sparse activations format to to matrix '''
    res = []
    for index, row in df.iterrows():
        acts = row['new_activations']
        result_vector = np.zeros(n_classes * 2)
        for act in acts:
            pos = act[0]
            value = act[1]
            result_vector[n_classes + pos] = value
        result_vector[n_classes:] = renorm(result_vector[n_classes:], alpha=1.0)
        result_vector[row['class_index']] = 1.0
        res.append(result_vector)
    
    res = np.array(res)
    return res
