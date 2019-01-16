import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import autosklearn.regression as auto
from joblib import dump

''' Loading train/test input '''
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

xtrain = np.c_[train_input_aa, train_input_q8]
xtest = np.c_[test_input_aa, test_input_q8]

''' Get train target '''
train_target_arrs = np.load('../train_output.npz')
means = []
stdev = []
for i in range(len(train_target_arrs)):
    means.append(train_target_arrs['arr_' + str(i)].mean())
    stdev.append(train_target_arrs['arr_' + str(i)].std())
means = np.array(means)
stdev = np.array(stdev)

''' Fit a regressor to the mean, stdev of the distance matrices '''
mean_regressor = auto.AutoSklearnRegressor()
mean_regressor.fit(xtrain, means)

std_regressor = auto.AutoSklearnRegressor()
std_regressor.fit(xtrain, stdev)

''' Predictions and saving '''
test_means = mean_regressor.predict(xtest)
np.savetxt('means.csv', test_means, delimiter=',')
test_stdev = std_regressor.predict(xtest)
np.savetxt('std.csv', test_stdev, delimiter=',')

dump(mean_regressor, 'mean_regressor.joblib')
dump(std_regressor, 'std_regressor.joblib')
