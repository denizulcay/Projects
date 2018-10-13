#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAME: Deniz Ulcay

This file contains a sequence to sequence Deep Learning prediction model I 
implemented to participate in a Kaggle competition on predicting secondary 
protein structure using cellular primary protein structure data.

This model uses three parallel Convolution layers to capture local information
and three consecutive bidirectional GRU layers to capture global information.
The output of the Convolution and bidirectional GRU layers are fed into a fully
connected layer followed by Softmax activation layer. Primary structure data
is represented as embedded vectors.
"""

import numpy as np 
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, concatenate, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
import csv

# The custom accuracy metric used for this task
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask),
                          tf.boolean_mask(y_, mask)), K.floatx())

# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

# prints the results
def print_results(x, y_, revsere_decoder_index):
    print(str(onehot_to_seq(y_, revsere_decoder_index).upper()))

# Computes and returns the n-grams of a particualr sequence, defaults to 
# trigrams
def seq2ngrams(seqs, n = 3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

# Prints out the test predictions to a CSV file
def print_csv(x_id, y_pred, revsere_decoder_index, file_name = 'test_pred.csv'):

    with open(file_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        row = ['id','expected']
        writer.writerow(row)
        writer.writerow('')
        
        for i in range(len(x_id)):
            y = str(onehot_to_seq(y_pred[i], revsere_decoder_index).upper())
            row = [x_id[i], y]
            writer.writerow(row)
            writer.writerow('')

    csvFile.close()
    
def load_data(maxlen_seq, file_train = 'train.csv', file_test = 'test.csv'):
    
    train_df = pd.read_csv(file_train)
    test_df = pd.read_csv(file_test)

    # Loading and converting the inputs to trigrams
    train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
    train_input_grams = seq2ngrams(train_input_seqs)

    # Same for test
    test_input_seqs = test_df['input'].values.T
    test_input_ids = test_df['id'].values.T
    test_input_grams = seq2ngrams(test_input_seqs)
    
    # Initializing and defining the tokenizer encoders and decoders based 
    # on the train set
    tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(train_input_grams)
    tokenizer_decoder = Tokenizer(char_level = True)
    tokenizer_decoder.fit_on_texts(train_target_seqs)
    
    # Using the tokenizer to encode and decode the sequences for use in 
    # training inputs
    train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
    train_input_data = sequence.pad_sequences(train_input_data,
                                              maxlen = maxlen_seq, 
                                              padding = 'post')
    
    # Targets
    train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
    train_target_data = sequence.pad_sequences(train_target_data, 
                                               maxlen = maxlen_seq, 
                                               padding = 'post')
    train_target_data = to_categorical(train_target_data)
    
    # Use the same tokenizer defined on train for tokenization of test
    test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
    test_input_data = sequence.pad_sequences(test_input_data, 
                                             maxlen = maxlen_seq, 
                                             padding = 'post')

    data = {}
    data["test_input_seqs"] = test_input_seqs
    data["test_input_ids"] = test_input_ids
    data["tokenizer_encoder"] = tokenizer_encoder
    data["tokenizer_decoder"] = tokenizer_decoder
    data["train_input_data"] = train_input_data
    data["train_target_data"] = train_target_data
    data["test_input_data"] = test_input_data
    
    return data


def build_model(maxlen_seq, n_words, n_tags):

    input = Input(shape = (maxlen_seq,))

    # An embedding layer mapping from words (n_words) to a vector of len 128
    x = Embedding(input_dim = n_words, output_dim = 128, 
                  input_length = maxlen_seq)(input)

    # Three parallel convolution layers to look for local correlations using 
    # the embedded representation of the inputs
    conv1 = Conv1D(filters=100, kernel_size=3, padding='SAME', strides=1, 
                   use_bias=True, activation='relu', name='conv1')(x)
    conv2 = Conv1D(filters=100, kernel_size=7, padding='SAME', strides=1, 
                   use_bias=True, activation='relu', name='conv2')(x)
    conv3 = Conv1D(filters=100, kernel_size=11, padding='SAME', strides=1, 
                   use_bias=True, activation='relu', name='conv3')(x)

    # Concatenation of the convolution outputs undergoes batch normalization
    concat12 = concatenate([conv1, conv2])
    conv_concat = concatenate([concat12, conv3])
    conv_norm = BatchNormalization()(conv_concat)

    # Three bidirectional GRUs to look for global correlations
    bgru1 = Bidirectional(GRU(units = 100, return_sequences = True, 
                              recurrent_dropout = 0.50, name='bgru1'))(conv_norm)
    bgru2 = Bidirectional(GRU(units = 100, return_sequences = True, 
                              recurrent_dropout = 0.50, name='bgru2'))(bgru1)
    bgru3 = Bidirectional(GRU(units = 100, return_sequences = True, 
                              recurrent_dropout = 0.50, name='bgru3'))(bgru2)

    # The output of the bidirectional GRUs is concatenated to the Convolution 
    # output to capture both local and global correlations in the input data
    bgru3_concat = concatenate([bgru3, conv_norm])

    # One fully connected layer followed by the softmax layer
    fc1 = TimeDistributed(Dense(64, activation = "relu", 
                                name="fc1"))(bgru3_concat)
    y = TimeDistributed(Dense(n_tags, activation = "softmax", 
                              name="softmax"))(fc1)

    model = Model(input, y)
    model.summary()

    # Compiles model with Adam optimizer, categorical cross entropy loss 
    # and the custom accuracy function as accuracy
    model.compile(optimizer = "adam", loss = "categorical_crossentropy",
                  metrics = ["accuracy", accuracy])

    return model


maxlen_seq = 754
data = load_data(maxlen_seq, 'train.csv', 'test.csv')

tokenizer_encoder = data["tokenizer_encoder"]
tokenizer_decoder = data["tokenizer_decoder"]
train_input = data["train_input_data"]
train_target = data["train_target_data"]

n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

X_train, X_val, y_train, y_val = train_test_split(train_input, train_target,
                                                  test_size = .1, 
                                                  random_state = 0)

# Using GPU to improve computation time
with tf.Session(config = tf.ConfigProto(log_device_placement = True)):
    
    # Build and fit the model
    model = build_model(maxlen_seq, n_words, n_tags)
    model.fit(X_train, y_train, batch_size = 128, epochs = 5, 
              validation_data = (X_val, y_val), verbose = 1)

    # Defining the decoders so that we can
    revsere_decoder_index = {value:key for key,value in 
                             tokenizer_decoder.word_index.items()}
    revsere_encoder_index = {value:key for key,value in 
                             tokenizer_encoder.word_index.items()}
        
    test_input = data["test_input_data"]
    test_seqs = data["test_input_seqs"]
    test_ids = data["test_input_ids"]

    y_test_pred = model.predict(test_input[:])
    
    print(len(test_input))
    for i in range(len(test_input)):
        print_results(test_seqs[i], y_test_pred[i], revsere_decoder_index)
    
    print_csv(test_ids, y_test_pred, revsere_decoder_index, 'test_pred.csv')
    model.save('my_model.h5')