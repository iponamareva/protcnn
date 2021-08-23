import json
import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from tqdm import tqdm

def read_data(data_path, partition):
  data = []
  for fn in os.listdir(os.path.join(data_path, partition)):
    with open(os.path.join(data_path, partition, fn)) as f:
      data.append(pd.read_csv(f, index_col=None))
  return pd.concat(data)


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

def load_all_files_in_dir(p, n_files=None) :
    df_list = []
    files = [f for f in listdir(p) if isfile(join(p, f))]
    print('Dir:', p, 'contains', len(files), 'files.')
    if n_files is not None:
        files = files[:n_files]
        print("Remained:", n_files, "files")
    
    for f in tqdm(files):
        filename = join(p, f)
        df = load_file_as_df(filename)
        df_list.append(df)

    df_conc = pd.concat(df_list)
    print("Files loaded and concatenated")
    return df_conc


def merge_and_preprocess(data_sample, df_conc, pos_dict=None):
    print("Merging and preprocessing...")
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

if __name__ == "__main__":
    kaggle_data_path = "random_split"
    google_data_path = "blundell_seed_ensemble_activations/"

    POS_FOR_CLASSES, classes_list = process_activations_dict()
    print("Number of classes in dictionary:", len(POS_FOR_CLASSES))

    for name in ["train"]:
        kaggle_df = read_data(kaggle_data_path, name)
        google_df = load_all_files_in_dir(google_data_path+name)
        res_df = merge_and_preprocess(kaggle_df, google_df)

        res_df = res_df.reset_index()
        res_df.to_csv("joined/"+name+".csv")


