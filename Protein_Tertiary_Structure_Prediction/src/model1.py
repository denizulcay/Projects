#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 01:09:28 2018

@author: denizulcay
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import GRU, Conv1D, Dropout, Activation, Bidirectional, BatchNormalization, Concatenate, TimeDistributed, Dense, Embedding, Layer, Reshape
from keras.optimizers import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.regularizers import l1, l2
import tensorflow as tf


''' Loading input data ''' 

def seq2ngrams(seqs, n = 3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

def seq2onehot(seq, n):
    out = np.zeros((len(seq), maxlen_seq, n))
    for i in range(len(seq)):
        for j in range(maxlen_seq):
            out[i, j, seq[i, j]] = 1
    return out

train_df = pd.read_csv('../train_input.csv')
test_df = pd.read_csv('../test_input.csv')

# Find maximum length sequence
num_train = len(train_df)
num_test = len(test_df)
maxlen_seq = max(train_df['length'].values.max(), test_df['length'].values.max())

# Loading and converting the inputs to ngrams
train_input_aa, train_input_q8 = train_df[['sequence', 'q8']].values.T
train_aa_grams = seq2ngrams(train_input_aa, n=3)
train_q8_grams = seq2ngrams(train_input_q8, n=3)

test_input_aa, test_input_q8 = test_df[['sequence', 'q8']].values.T
test_aa_grams = seq2ngrams(test_input_aa, n=3)
test_q8_grams = seq2ngrams(test_input_q8, n=3)

# Initializing and defining the tokenizer encoders based on the train set
tokenizer_encoder_aa = Tokenizer()
tokenizer_encoder_aa.fit_on_texts(train_aa_grams)
tokenizer_encoder_q8 = Tokenizer()
tokenizer_encoder_q8.fit_on_texts(train_q8_grams)

# Using the tokenizer to encode the input sequences for use in training and testing
train_input_aa = tokenizer_encoder_aa.texts_to_sequences(train_aa_grams)
train_input_aa = sequence.pad_sequences(train_input_aa, maxlen = maxlen_seq, padding = 'post', truncating='post')
train_input_q8 = tokenizer_encoder_q8.texts_to_sequences(train_q8_grams)
train_input_q8 = sequence.pad_sequences(train_input_q8, maxlen = maxlen_seq, padding = 'post', truncating='post')

test_input_aa = tokenizer_encoder_aa.texts_to_sequences(test_aa_grams)
test_input_aa = sequence.pad_sequences(test_input_aa, maxlen = maxlen_seq, padding = 'post', truncating='post')
test_input_q8 = tokenizer_encoder_q8.texts_to_sequences(test_q8_grams)
test_input_q8 = sequence.pad_sequences(test_input_q8, maxlen = maxlen_seq, padding = 'post', truncating='post')

n_words_aa = len(tokenizer_encoder_aa.word_index) + 1
n_words_q8 = len(tokenizer_encoder_q8.word_index) + 1

train_input_aa_hot = seq2onehot(train_input_aa, n_words_aa)
train_input_q8_hot = seq2onehot(train_input_q8, n_words_q8)

test_input_aa_hot = seq2onehot(test_input_aa, n_words_aa)
test_input_q8_hot = seq2onehot(test_input_q8, n_words_q8)


''' Loading training target data '''
train_target_arrs = np.load('../train_output.npz')

seq_len = train_df['length'].values
train_target_data = np.zeros((num_train, maxlen_seq, maxlen_seq))
for i in range(num_train):
    train_target_data[i, :seq_len[i], :seq_len[i]] = train_target_arrs['arr_' + str(i)]
    

X_train = [train_input_aa[:4000], train_input_q8[:4000], train_input_aa_hot[:4000], train_input_q8_hot[:4000]]
y_train = train_target_data[:4000]

X_val = [train_input_aa[4000:], train_input_q8[4000:], train_input_aa_hot[4000:], train_input_q8_hot[4000:]]
y_val = train_target_data[4000:]


# Define custom distance layer for making predictions
class Distance(Layer):
    def __init__(self, **kwargs):
        super(Distance, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Distance, self).build(input_shape)

    def call(self, x):
        r = tf.reduce_sum(x*x, 2)
        r = tf.expand_dims(r, -1)
        xx = tf.einsum('bij,bjk->bik', x, tf.linalg.transpose(x))
        D = r - 2*xx + tf.linalg.transpose(r)
        return D
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])
    
###### Model ######

def conv_block(x):

    cnn = Conv1D(64, 11, padding="same")(x)
    cnn = TimeDistributed(Activation("relu"))(cnn)
    cnn = TimeDistributed(BatchNormalization())(cnn)
    cnn = TimeDistributed(Dropout(0.5))(cnn)
    cnn = Concatenate(-1)([x, cnn])
    
    return cnn

def super_conv_block(x):
        
    c3 = Conv1D(32, 1, padding="same")(x)
    c3 = TimeDistributed(Activation("relu"))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)
    
    c7 = Conv1D(64, 3, padding="same")(x)
    c7 = TimeDistributed(Activation("relu"))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)
    
    c11 = Conv1D(128, 5, padding="same")(x)
    c11 = TimeDistributed(Activation("relu"))(c11)
    c11 = TimeDistributed(BatchNormalization())(c11)
    
    x = Concatenate(-1)([x, c3, c7, c11])
    x = TimeDistributed(Dropout(0.5))(x)
        
    return x
    
def model1():
    
    input_aa = Input(shape = (maxlen_seq, ))
    input_q8 = Input(shape = (maxlen_seq, ))
    
    # one-hot representations of input
    input_aa_hot = Input(shape = (maxlen_seq, n_words_aa))
    input_q8_hot = Input(shape = (maxlen_seq, n_words_q8))
    
    embed_aa = Embedding(input_dim = n_words_aa, output_dim = 64, input_length = maxlen_seq)(input_aa)
    embed_q8 = Embedding(input_dim = n_words_q8, output_dim = 64, input_length = maxlen_seq)(input_q8)
    
    merge_in = Concatenate(-1)([input_aa_hot, embed_aa, embed_q8, input_q8_hot])
   
    x = super_conv_block(merge_in)
    x = conv_block(x)
    x = conv_block(x)
    x = conv_block(x)
    
    x = Bidirectional(GRU(units = 256, return_sequences = True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation = "relu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    
    y = Distance()(x)
    
    model = Model([input_aa, input_q8, input_aa_hot, input_q8_hot], y)
    
    return model


model = model1()
model.summary()


model.compile(    
    optimizer='Nadam',
    loss = 'mean_squared_error')
#with tf.Session( config = tf.ConfigProto( log_device_placement = True ) ):
model.fit( X_train, y_train,
          batch_size = 64, epochs = 100,
          validation_data = (X_val, y_val),
          callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)],
          verbose = 1)


''' save model '''
model.save('model1test_1.h5')
model.save_weights('model1test_1_weights.h5')