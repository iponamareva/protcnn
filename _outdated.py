class DataGenerator(Sequence):
    def __init__(self, mode, batch_size=256, dim=(100, 21), shuffle=True):

        print('Initializing data generator in mode', mode)
        if mode=='train':
            kaggle_df = read_data(data_path=kaggle_data_path, partition=mode)
            google_df = load_all_files_in_dir(google_data_path+mode)
            self.df = merge_and_preprocess(kaggle_df, google_df)
            self.df = self.df[:1000]
            print(len(self.df))
        else:
            print("Error: unknown mode")

        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.df)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Select and process data
        df_batch = self.df.loc[indexes]

        X = encode_and_pad(df_batch)
        y = expand_sparse_activations_from_df(df_batch, alpha=1.0)

        return (X, y)
