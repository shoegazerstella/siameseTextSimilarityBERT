import pandas as pd
from collections import Counter
import numpy as np
import random
import pickle
import scipy
import time, os

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from model import siamese_bilstm_model
#from model2 import siamese_cnn_model

from generator import DataGenerator
from embedding import load_embedding_model, embed_batch

DATA_DIR = 'data'


def load_dataset():

    df = pd.read_csv( os.path.join(DATA_DIR, 'sample_data.csv'), sep=',')

    train, dev = train_test_split(df, test_size=0.1)
    dev, test = train_test_split(dev, test_size=0.1)

    train = train.reset_index(drop=True)
    dev = dev.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(train.shape, dev.shape, test.shape)
    return train, dev, test


def _test(best_model_path, test, sentence_bert_model):

    model = load_model(best_model_path)

    X_test_1, X_test_2, Y_test = embed_batch(test, sentence_bert_model)

    preds = list( model.predict([X_test_1, X_test_2], verbose=1).ravel() )

    results_round = [np.round(i) for i in preds]

    print('ACCURACY', np.around(accuracy_score(Y_test, results_round),2))
    print(classification_report(Y_test, results_round, target_names=['0', '1']))
    print(confusion_matrix(Y_test, results_round))

    return


def run():

    train, dev, test = load_dataset()
    
    model = siamese_bilstm_model()
    #model = siamese_cnn_model()

    # load sentence embedding model
    sentence_bert_model = load_embedding_model()

    BATCH_SIZE = 128
    EPOCHS = 30

    # Generators
    training_generator = DataGenerator(train, sentence_bert_model, batch_size=BATCH_SIZE)
    validation_generator = DataGenerator(dev, sentence_bert_model, batch_size=BATCH_SIZE)

    # CALLBACKS
    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
    
    checkpoint_dir = os.path.join(DATA_DIR, 'checkpoints_bert_lstm/', str(int(time.time())) + '/')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    best_model_path = checkpoint_dir + 'model.h5'
    model_checkpoint = ModelCheckpoint(best_model_path,
                                        monitor='loss',
                                        save_best_only=True, 
                                        save_weights_only=False, 
                                        verbose=1)

    # Train model on dataset
    model.fit(training_generator,
            validation_data=validation_generator,
            validation_steps=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[early_stopping, model_checkpoint])

    _test(best_model_path, test, sentence_bert_model)

    return 

if __name__ == "__main__":
    run()