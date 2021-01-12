import numpy as np
import keras
from embedding import embed_batch

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, sentence_bert_model, batch_size=10, shuffle=True):
        'Initialization'
        self.df = df
        self.list_IDs = list(self.df.index)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.sentence_bert_model = sentence_bert_model

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        df_batch = self.df.iloc[list_IDs_temp]

        # Generate data
        X1, X2, labels = embed_batch(df_batch, self.sentence_bert_model)

        return [X1, X2], labels