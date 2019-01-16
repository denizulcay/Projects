#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the implementation of our end-to-end architecture with BLAST
alignment distances. The functions that are commented out are some of the 
sequence models and layers we later decided not to use. These are discussed
in the final report.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
import Levenshtein as lev

from helperfunctions import *


''' Loading input data '''
def seq2ngrams(seqs, n = 3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

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

''' Loading training target data '''
train_target_arrs = np.load('../train_output.npz')

seq_len = train_df['length'].values
train_target_data = np.zeros((num_train, maxlen_seq, maxlen_seq))

for i in range(num_train):
    train_target_data[i, :seq_len[i], :seq_len[i]] = train_target_arrs['arr_' + str(i)]

# Loading sequence alignments for training and test data and stitching 
# corresponding distances in the training target data onto 
# [maxlen_seq x maxlen_seq] matrices
stitches_aa = []
for i in range(len(train_df)):

    alignments_aa = get_alignments('../blast-train/matches/aa_match'+str(i)+'.txt')
    stitches_aa.append(get_stitches_tr(alignments_aa, train_df, 
                                       train_target_data, maxlen_seq, i))
    
train_stitches_aa = np.array(stitches_aa).reshape((-1, maxlen_seq, maxlen_seq))

alignments_aa = get_alignments('../blast-train/matches/aa_match_te.txt')
test_stitches_aa = get_stitches_te(alignments_aa, train_df, train_target_data,
                                   maxlen_seq, test_df)


''' model '''
# this is done to force the symmetry of the distance matrix
class Symmetrize(Layer):
    def __init__(self, **kwargs):
        super(Symmetrize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Symmetrize, self).build(input_shape)

    def call(self, x):
        x = (x+tf.linalg.transpose(x))/2
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])


''' DO NOTE USE: This produced bad results, see writeup
# Define custom distance layer for making predictions
class Distance(Layer):
    def __init__(self, **kwargs):
        super(Distance, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Distance, self).build(input_shape)

    def call(self, x):
        # We want: matrix D s.t. D[i,j] = (x[i]-x[j])(x[i]-x[j])'
        # Rewrite: D[i,j] = r[i] - 2 a[i]a[j]' + r[j]
        # where r[i] is the squared norm of x[i]
        r = tf.reduce_sum(x*x, 2)
        r = tf.expand_dims(r, -1)
        # computing center term across batch 
        xx = tf.einsum('bij,bjk->bik', x, tf.linalg.transpose(x))
        D = r - 2*xx + tf.linalg.transpose(r)
        return D

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])
'''

''' DO NOT USE: The validation error diverges after epoch 2 or 3
def model2(emb):
    droprate = 0.25

    def conv_block(x, n_channels, droprate):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
        x = Dropout(droprate)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
        return x

    def up_block(x, n_channels):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling1D(size = 2)(x)
        x = Conv1D(n_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
        return x

    conv_input = Conv1D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(emb)

    conv1 = conv_block(conv_input, 128, droprate)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = conv_block(pool1, 192, droprate)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = conv_block(pool2, 384, droprate)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = conv_block(pool3, 768, droprate)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = conv_block(pool4, 1536, droprate)

    up4 = up_block(conv5, 768)
    up4 = concatenate([conv4,up4], axis = 2)
    up4 = conv_block(up4, 768, droprate)

    up3 = up_block(up4, 384)
    up3 = concatenate([conv3,up3], axis = 2)
    up3 = conv_block(up3, 384, droprate)

    up2 = up_block(up3, 192)
    up2 = ZeroPadding1D(padding=(0,1))(up2)
    up2 = concatenate([conv2,up2], axis = 2)
    up2 = conv_block(up2, 192, droprate)

    up1 = up_block(up2, 128)
    up1 = ZeroPadding1D(padding=(0,1))(up1)
    up1 = concatenate([conv1,up1], axis = 2)
    up1 = conv_block(up1, 128, droprate)

    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)
    return up1
    
# This is a dilated convolution block that iteratively increases dilation to
# capture both local and global correlations
def conv_d(x):
    # Defining dilated convolutional blocks   
    def dilated_block(x, filter_size, dilation):
        cnn = Conv1D(64, filter_size, padding="same", activation="relu", dilation_rate=dilation)(cnn)
        cnn = Dropout(0.1)(cnn)
    
        return cnn
    
    x = dilated_block(x, 3, 1)
    x = dilated_block(x, 3, 2)
    x = dilated_block(x, 3, 4)
    x = dilated_block(x, 3, 8)
    x = dilated_block(x, 3, 16)
    x = dilated_block(x, 3, 1)
    x = Conv1D(64, 1, padding="same")(x)
    cnn = BatchNormalization()(x)
    
    return cnn
'''

## This is the convolutional network with bidirectional GRUs
def model4(emb):
    # defining convolutional blocks
    def conv(x, filter_size):
        x = ZeroPadding2D((filter_size//2, 0), data_format='channels_first')(x)
        x = Conv2D(filters=64, kernel_size=(filter_size, 128), data_format='channels_first', activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        return x
    
    emb = Reshape([1, maxlen_seq, 128])(emb)

    # Defining 3 convolutional layers with different kernel sizes
    conv1 = conv(emb, 3)
    conv2 = conv(emb, 7)
    conv3 = conv(emb, 11)
    conv_ = Concatenate(-1)([conv1, conv2, conv3])
    conv_ = Permute((2, 1, 3))(conv_)
    conv_ = Reshape([maxlen_seq, 3*64])(conv_)

    # Defining 3 bidirectional GRU layers; taking the concatenation of outputs
    gru1 = Bidirectional(GRU(64, return_sequences='True', recurrent_dropout=0.1))(conv_)
    gru2 = Bidirectional(GRU(64, return_sequences='True', recurrent_dropout=0.1))(gru1)
    gru3 = Bidirectional(GRU(64, return_sequences='True', recurrent_dropout=0.1))(gru2)
    comb = Concatenate(-1)([gru1, gru2, gru3, conv_])
    return comb

# This returns the model for tertiary structure prediction
def get_model(num):
    
    input_aa = Input(shape = (maxlen_seq, ))
    input_q8 = Input(shape = (maxlen_seq, ))
    input_stitches = Input(shape = (maxlen_seq, maxlen_seq))
    
    # Embedding aa and q8 inputs (both kinds of sequences)
    embed_aa = Embedding(input_dim = n_words_aa, output_dim = 128, input_length = maxlen_seq)(input_aa)
    embed_q8 = Embedding(input_dim = n_words_q8, output_dim = 128, input_length = maxlen_seq)(input_q8)

    # use known good models to extract features from each of these sequences
    aa = model4(embed_aa)
    q8 = model4(embed_q8)

    # concatenate these features
    x = Concatenate(-1)([aa, q8, input_stitches])

    # Some fully connected layers before distance calculation
    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(maxlen_seq))(x)

    y = Symmetrize()(x)

    # Defining the model as a whole and printing the summary
    return Model([input_aa, input_q8, input_stitches], y)


model = get_model(4)
optim = Nadam()

''' fit model '''
model.summary()
model.compile(optimizer = optim, loss = 'mean_squared_error')

train_aa = train_input_aa[:4000]
val_aa = train_input_aa[4000:]

train_q8 = train_input_q8[:4000]
val_q8 = train_input_q8[4000:]

train_tar = train_target_data[:4000]
val_tar = train_target_data[4000:]

train_stitches = train_stitches_aa[:4000]
val_stitches = train_stitches_aa[4000:]

model.fit([train_aa, train_q8, train_stitches], train_tar,
          batch_size = 64, epochs = 100,
          validation_data = ([val_aa, val_q8, val_stitches], val_tar),
          callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)],
          verbose = 1)

''' Save model '''
model.save('seq2mat_9.h5')
model.save_weights('seq2mat_9_weights.h5')

''' Predictions '''
pred = model.predict([test_input_aa, test_input_q8, test_stitches_aa])
# saving
out = []
for i in range(len(pred)):
    l = test_df['length'][i]
    pred_ = pred[i, :l, :l]
    out.append(pred_)
np.savez('test9.npz', *out)