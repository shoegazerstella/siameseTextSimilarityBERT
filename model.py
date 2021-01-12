import time, os
import pickle
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model


def siamese_bilstm_model(shape=(768, 1)):
    
    number_lstm_units = 100
    rate_drop_dense = 0.25
    number_dense_units = 50
    activation_function = 'relu'
    rate_drop_lstm =  0.17

    # define inputs
    input1 = tf.keras.Input(shape=shape)
    input2 = tf.keras.Input(shape=shape)

    # Creating LSTM Encoder
    lstm_layer = Bidirectional(LSTM(number_lstm_units, 
                                    dropout=rate_drop_lstm, 
                                    recurrent_dropout=rate_drop_lstm))

    x1 = lstm_layer(input1)
    x2 = lstm_layer(input2)
    merged = concatenate([x1, x2])
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)

    merged = Dense(number_dense_units, activation=activation_function)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[input1, input2], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #nadam
    
    model.summary()
    
    return model