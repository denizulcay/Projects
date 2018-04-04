#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deniz Ulcay - du2147

This file contains a perceptron algorithm that uses different ngram models to classify spam emails.
The performances of different models are compared to determine the best one with different training
data sizes.
"""

import sys
import csv
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import math
import random
import os
import os.path
import numpy as np
import time as t

def corpus_reader(corpusfile, limit = 1000):    # yields: {'Y': label (str), 'X': sequence (list)} (dict)
# Reads the corpus from a csv file
    with open(corpusfile,'r') as corpus_file:
        corpus = csv.reader(corpus_file)
        
        i = 0
        
        for line in corpus:
            
            label = line[0]
            
            if label == '0':
                label = -1
            elif label == '1':
                label = 1
                
            text = line[1]
            
            if i > limit:
                break
            
            if label == 'label':
                continue            
            if text.strip():
                sequence = text.lower().strip().split()
                data = {'Y': label, 'X': sequence}
                yield data
            
            i += 1

def get_lexicon(corpus):        # returns: (lexicon (set), bigrams (set), documents that contain word (dict),  no_of_docs (int)) (tuple)
# extracts the features from the textual data
    word_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    
    doc_count = 0
    
    for document in corpus:
        
        doc_count += 1
        word_seen = set([])
        label = document['Y']
        text = document['X']
        
        for i in range(len(text)): 
            
            if text[i] not in word_seen:
                
                word_counts[text[i]] += 1
                word_seen.add(text[i])
            
            if not i == 0:
                
                bigram = tuple(text[i - 1: i + 1])
                bigram_counts[bigram] += 1
            
    lexicon = set(word for word in word_counts)
    bigrams = set(bigram for bigram in bigram_counts)
    
    return (lexicon, bigrams, word_counts, doc_count)


def get_data(sequence, lexicon=None, bigrams=None):      # returns: data_point (list)
# if lexicon or bigrams, does not return tuples not in training
    unigram_c = defaultdict(int)                         
    bigram_c = defaultdict(int)
                                                            
    i = 0
    
    for word in sequence:   
        
        if word in lexicon:
            unigram_c[word] += 1

        if i != 0:
            
            ngram = sequence[i - 1: i + 1]
            ngram = tuple(ngram)
            
            if ngram in bigrams:
                bigram_c[ngram] += 1
        
        i += 1
        
    uni_point = [unigram_c[word] if word in unigram_c else 0 for word in lexicon]
    bi_point = [bigram_c[word] if word in bigram_c else 0 for word in bigrams]
    
    return (uni_point, bi_point)


class Perceptron(object):
    
    def __init__(self, corpusfile, limit):
        
        self.limit = limit
        generator = corpus_reader(corpusfile, self.limit)
        
        print("getting lexicon...")
        self.lexicon_s, self.bigrams_s, self.word_in_docs, self.doc_count = get_lexicon(generator)
        
        self.lexicon = list(self.lexicon_s)
        self.bigrams = list(self.bigrams_s)
        
        print("count ngrams...")
        generator = corpus_reader(corpusfile, self.limit)
        self.unigram_train, self.bigram_train = self.count_ngrams(generator)
        
        print("preprocess...")
        self.unigram_train = self.preprocess(self.unigram_train)
        self.bigram_train = self.preprocess(self.bigram_train)
        
        print("inverse freq...")
        self.inverse_freq_train = self.inverse_freq_data(self.unigram_train)
        
        print("lifting...")
        self.lexicon_l = self.lexicon[:]
        self.bigrams_l = self.bigrams[:]
        
        self.unigram_train = self.data_lift(self.unigram_train)
        self.lexicon_l.append("UNK")
        
        self.bigram_train = self.data_lift(self.bigram_train)
        self.bigrams_l.append("UNK")

        print("training...")
        self.train()
        self.online_to_batch()
        
        
    def test(self, testfile, limit):
        
        generator = corpus_reader(testfile, limit)
        unigram_test, bigram_test = self.count_ngrams(generator)
        
        unigram_test = self.preprocess(unigram_test)
        bigram_test = self.preprocess(bigram_test)
        
        inverse_freq_test = self.inverse_freq_data(unigram_test)
        
        unigram_test = self.data_lift(unigram_test)
        bigram_test = self.data_lift(bigram_test)

        uni_accuracy = self.perceptron_test(unigram_test, self.lexicon_l, self.uni_final)
        bi_accuracy = self.perceptron_test(bigram_test, self.bigrams_l, self.bi_final)
        inv_accuracy = self.perceptron_test(inverse_freq_test, self.lexicon_l, self.inv_final)
        
        return (uni_accuracy, bi_accuracy, inv_accuracy)
    
    
    def perceptron_test(self, data, features, classifier):
        
        labels = data['Y']
        labels = np.hstack(labels)       
        
        counts = data['X']
        score = 0
        
        for i in range(len(counts)):
            
            point = counts[i]
            f = sum(classifier * point)
            label = labels[i]

            if (f * label) >= 0:
                score += 1
        
        accuracy = score / len(counts)
        return accuracy
        
        
    def train(self):
        
        uni_classifiers = []
        bi_classifiers = []
        inv_freq_classifiers = []
        
        first = True
        
        for i in range(2):
            
            unigram_train = self.random_data_shuffle(self.unigram_train)
            bigram_train = self.random_data_shuffle(self.bigram_train)
            inv_freq_train = self.random_data_shuffle(self.inverse_freq_train)
            print(len(self.lexicon))
            print(len(self.lexicon_l))
            print(unigram_train['X'].shape)            
            uni_classifiers += self.perceptron_train(unigram_train, self.lexicon_l, first)
            bi_classifiers += self.perceptron_train(bigram_train, self.bigrams_l, first)
            inv_freq_classifiers += self.perceptron_train(inv_freq_train, self.lexicon_l, first)
            
            first = False
        
        self.uni_classifiers = np.vstack((uni_classifiers[:]))
        self.bi_classifiers = np.vstack((bi_classifiers[:]))
        self.inv_freq_classifiers = np.vstack((inv_freq_classifiers[:]))
        
        
    def online_to_batch(self):
    # calculates the batch weights for the different models
        i_uni = len(self.unigram_train)
        self.uni_final = np.sum(self.uni_classifiers[i_uni:], axis=0) / (i_uni + 1)
        
        i_bi = len(self.bigram_train)
        self.bi_final = np.sum(self.bi_classifiers[i_bi:], axis=0) / (i_bi + 1)
        
        i_inv = len(self.inverse_freq_train)
        self.inv_final = np.sum(self.inv_freq_classifiers[i_inv:], axis=0) / (i_inv + 1)
        
        
    def perceptron_train(self, data, features, first=True):
    # training algorithm of the perceptron
        labels = data['Y']
        labels = np.hstack(labels)
        
        counts = data['X']
        
        w_i = np.zeros(len(features))
        
        if first:
            classifiers = [w_i]
        else:
            classifiers = []
        
        for i in range(len(counts)):
            
            point = counts[i]
            label = labels[i]

            f = sum(w_i * point)
            
            if (f * label) <= 0:
                w_i = w_i + label * point
            
            classifiers.append(w_i)
            
        return classifiers
        
    
    def count_ngrams(self, corpus):         # returns: (unigrams_dataset (dict), bigrams_dataset (dict)) (tuple)
    # creates the data matrix for unigram and bigram representations            
        unigram_labels = []
        unigram_data = []
        
        bigram_labels = []
        bigram_data = []
        
        i = 0
        doc_count = 0
        
        for document in corpus:
            
            i += 1
            
            if not i % 10:
                print("{}/{}".format(i, self.doc_count))
            
            label = document['Y']
            text = document['X']
            
            unigram_labels.append(label)
            bigram_labels.append(label)
            
            uni, bi = get_data(text, lexicon = self.lexicon_s, bigrams=self.bigrams_s)
            
            unigram_data += uni
            bigram_data += bi
            
            doc_count+= 1
            
        unigram_matrix = np.reshape(unigram_data, newshape=(doc_count, len(self.lexicon_s)))    
        bigram_matrix = np.reshape(bigram_data, newshape=(doc_count, len(self.bigrams_s)))
        
        unigram_labels = np.array(unigram_labels).reshape((doc_count, 1))
        bigram_labels = np.array(bigram_labels).reshape((doc_count, 1))
        
        unigrams = {"Y": unigram_labels, "X": unigram_matrix}
        bigrams = {"Y": bigram_labels, "X": bigram_matrix}
        
        return (unigrams, bigrams)


    def inverse_freq_data(self, data):
    # adds the inverse frequencies for inverse frequency representation
        labels = data['Y']
        texts = data['X']
        texts_t = np.transpose(texts)
        
        texts_t_inv = [np.multiply(texts_t[self.lexicon.index(i)], math.log( (self.doc_count / self.word_in_docs[i]), 10)) for i in self.lexicon_s]
        data['X'] = np.transpose(texts_t_inv)
        
        return data
        
    
    def data_lift(self, data):
    # adds bias to the data
        counts = data['X']
        labels = data['Y']
        
        bias = np.vstack(np.ones(len(counts)))
        
        data['X'] = np.hstack((counts, bias))
        
        return data
        
        
    def preprocess(self, data):     # returns: data (dict)
        
        counts = data['X']
        labels = data['Y']
        
        counts_t = np.transpose(counts)
        
        variance = [(i, np.mean(counts_t[i]), np.var(counts_t[i])) for i in range(len(counts_t))]
        
        counts_tnorm = [(np.subtract(counts_t[i[0]], i[1]))/1 for i in variance]
        counts_norm = np.transpose(counts_tnorm)
        
        data['X'] = counts_norm
        
        return data
    
    
    def random_data_shuffle(self, data):    # returns: data (dict)
        
        counts = data['X']
        labels = data['Y']
        
        stack = np.hstack((labels, counts))
        np.random.shuffle(stack)
        
        labels = np.vstack(stack[:, 0]).astype(float)
        counts = stack[:,1:].astype(float)
        
        data = {'Y': labels, 'X': counts}
        
        return data
    
    def hall_of_fame(self):
        
        weights = [(self.uni_final[i], self.lexicon_l[i]) for i in range(len(self.lexicon_l))]
        weights = sorted(weights, key=lambda x:x[0], reverse=True)
        top_ten = weights[1:11]
        bot_ten = weights[-10:]
        
        return (top_ten, bot_ten)
    
if __name__ == "__main__":
    
    sizes = [2000]
    split = [0.4, 0.6, 0.75, 0.8, 0.9]
    uni_accuracies = []
    bi_accuracies = []
    inv_accuracies = []
    
    for size in sizes:
        
        for sp in split:
            train_size = size * sp
            decepticon = Perceptron(sys.argv[1], train_size)
                
            test_size = size * (1 - sp)
            uni_accuracy, bi_accuracy, inv_accuracy = decepticon.test(sys.argv[2], test_size)
            top_ten, bot_ten = decepticon.hall_of_fame()
        
            uni_accuracies.append(uni_accuracy)
            bi_accuracies.append(bi_accuracy)
            inv_accuracies.append(inv_accuracy)
    
    print("Unigram Accuracy: {}".format(uni_accuracy))
    print("Bigram Accuracy: {}".format(bi_accuracy))
    print("Inv Accuracy: {}".format(inv_accuracy))
    print(top_ten)
    print(bot_ten)
    

    uplot, = plt.plot(split, uni_accuracies, 'g--')
    biplot, = plt.plot(split, bi_accuracies, 'y')
    iplot, = plt.plot(split, inv_accuracies, 'r')

    plt.ylabel('Accuracy')
    plt.xlabel('Data Split')
    plt.title('Accuracy vs Data Split')
    plt.legend([uplot, biplot, iplot], ['green: unigram', 'yellow: bigram', 'red: inverse frequency'])
    plt.show()
    
    
