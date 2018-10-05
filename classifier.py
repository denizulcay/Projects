#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:21:50 2018

@author: denizulcay
"""

import random
import numpy as np
import scipy
import time
import sys
from scipy.io import loadmat


class Util:
    ''' Common functions I may want to use in the future'''
    def getGaussianProbability(x, mean, covariance):
        diff = np.array(np.subtract(x,mean))
        covDet = np.linalg.det(covariance)
        if (covDet == 0):
            covDet = np.exp(-700) #this is small enough.
        invCov = np.linalg.inv(covariance)
        exponent = - np.matmul(np.transpose(diff), np.matmul(invCov, diff))
        returnval = np.exp(exponent)/(np.sqrt(2 * scipy.pi * covDet))
        return returnval

    def getNormDistanceMatrix(data, x, norm):
        #returns the normed distance in each dimension.
        data2 = data - x;
        data2 = np.absolute(data2)
        return np.power(data2, norm)

    def getWeightedNormDistanceMatrix(normedDistanceMatrix, weight):
        #normed distance matrix:
        #x1 [........]
        #x2 [........]
        #...
        #returns a vector of weighted distance for each vector.
        return np.matmul(normedDistanceMatrix, np.transpose([weight]))

    def getNormDistanceVector(v, x, norm):
        #just for clarification on when I am using this.
        return Util.getNormDistanceMatrix(v, x, norm)

    def getWeightedNormDistanceVector(v, weight):
        #returns a number.
        return np.matmul(v, np.transpose[weight])

    def getDistanceVector(data, x, norm, weight):
        ''' Given matrix of vectors, return the vector of their normed distances'''
        normeddistanceMatrix = Util.getNormDistanceMatrix(data, x, norm)
        return Util.getWeightedNormDistanceMatrix(normeddistanceMatrix, weight)
    
    def getDistance(v, x, norm, weight):
        ''' Given a vector , return its normed distance. '''
        normeddistancevector = Util.getNormDistanceVector(v, x, norm)
        return Util.getWeightedNormDistanceVector(normeddistancevector, weight)

class Classifier:
    def __init__(self, data, trainPercentages, runs, dimensions):
        self.data = data
        self.trainPercentages = trainPercentages

        self.runs = runs
        self.accuracy= 0
        self.dimensions = dimensions
        self.bestDimension = 0
        self.bestTrainPercentage = 0.5
        self.classificationSpeed = 0

        self.dimensionsSelected = []
        return

    def getAccuracy(self):
        return self.accuracy

    def getBestDimension(self):
        return self.bestDimension

    def getBestTrainPercentage(self):
        return self.bestTrainPercentage

    def run(self):
        for trainPercent in self.trainPercentages:
            for dimension in self.dimensions:
                for i in range(0, runs):
                    train, test = self.randomlySelectTrainData(trainPercent)
                    print("Training...")
                    training_accuracy = self.train(train, dimension)
                    print("Testing...")
                    test_accuracy = self.test(test)
                    print("training accuracy: {}, test_accuracy: {}, dimension:{}, percentageOfTrainData:{}, classificationSpeed:{}".format(
                          training_accuracy,      test_accuracy,     dimension, trainPercent, self.classificationSpeed))
                    if (test_accuracy > self.accuracy):
                        self.accuracy = test_accuracy
                        self.bestDimension = dimension
                        self.bestTrainPercentage = trainPercent

    def randomlySelectTrainData(self, trainPercent):
        data = self.data['X']
        label = self.data['Y']
        data_size = len(data) -1
        trainDataSize = int(data_size * trainPercent)
        selection = [random.randint(0,data_size) for i in range(0,trainDataSize)]
        absent = [ i for i in range(0, data_size) if i not in selection]

        trainData  = [ data[i] for i in selection]
        trainLabel = [label[i] for i in selection]
        train = {'X' : trainData, 'Y':trainLabel}

        testData = [data[i] for i in absent]
        testLabel = [label[i] for i in absent]
        test = {'X': testData, 'Y' : testLabel}
        return (train,test)

    def selectTopDimensions(self, data):
        #here we populate dimensions selected.
        a = len(data[0]) #total dims available.
        data2 = np.transpose(data)
        variance = [ (i, np.mean(data2[i]), np.var(data2[i])) for i in range(0, len(data2)) ]
        variance = sorted(variance, key=lambda x: x[2], reverse=True)
        self.dimensionsSelected = [variance[i] for i in range(0, self.dimension)]
        return;

    def preprocess(self, data):
        def normalize(array, tup):
            #given a tuple containing  x, mean and variance normalize this array
            return  (np.subtract(array, tup[1]))/np.sqrt(tup[2])
        #select dimensions with have already been selected.
        data2 = np.transpose(data)
        data2Normalized = [ normalize(data2[i[0]] , i) for i in self.dimensionsSelected]
        return np.transpose(data2Normalized)

    def train(self, train, dimension):
        self.perClassInfo = {}
        self.dimensionsSelected=[]
        self.dimension = dimension

        #let's preprocess to make it easier for mle classifier.
        olddata = train['X']
        self.selectTopDimensions(olddata) # select dimensions and store selected dimensions.
        train['X'] = self.preprocess(olddata)
        self.learn(train)
        return self.test(train, preprocess = False) #data has already been preprocessed.


    def test(self, test, preprocess = True):
        t = time.time()
        data = test['X']
        label = test['Y']
        if (preprocess):
            data = self.preprocess(data)
        correctLabel = 0
        l =   [label[i][0] == self.classify(data[i]) for i in range(len(data))]
        result = sum(l)/len(l)
        self.classificationSpeed = (time.time() - t)/len(data)
        return result

    def classify(self, x):
        ''' This is to be overridden'''
        return 0
    def learn(self, train):
        '''This is to be overridden'''
        return 0

class MLEClassifier(Classifier):
    def __init__(self, data, trainPercentages, runs, dimensions):
        super(MLEClassifier, self).__init__(data, trainPercentages, runs, dimensions)
        self.perClassInfo = {}
        self.dimension = 0
        return

    def analyzeOneClass(self, data, totalSample):
        prob = len(data)/totalSample
        mean = []
        covariance = []
        mean = sum(data)/len(data)
        diff = data - mean
        covariance = np.divide(np.matmul(np.transpose(diff), diff), len(data)) + 0.01 * np.identity(len(mean))
        det = np.linalg.det(covariance)
        #Also we have already analyzed that MLE estimates are biased
        #when we don't know the mean. The bias is by N-1/N therefore, we need
        #to multilply the coveraince result by N/N-1 to unbias it a bit
        unbiasFactor = len(data) / (len(data)-1)
        covariance = np.multiply(covariance, unbiasFactor)
        return (mean, covariance, prob)

    def learn(self, train):
        data =  train['X']
        label = train['Y']
        for i in range(0, len(label)):
            label_i = label[i][0]
            if label_i not in self.perClassInfo:
                i_data = [data[j] for j in range(i, len(label)) if label[j][0] == label_i]
                mean,variance,prob = self.analyzeOneClass(i_data, len(data))
                self.perClassInfo[label_i] = (mean, variance, prob)

    def classify(self, x):
        maxprob = 0
        maxprobkey = 0
        for key in self.perClassInfo.keys():
            (mean, variance, classprob) = self.perClassInfo[key]
            prob = Util.getGaussianProbability(x, mean, variance) * classprob
            if prob > maxprob:
                maxprob = prob
                maxprobkey = key
        return maxprobkey

class KNNClassifier(Classifier):
    def __init__(self, data, trainPercentages, runs, ks, dimensions, norm=2):
        super(KNNClassifier, self).__init__(data,trainPercentages, runs, dimensions)
        self.possibleks = ks
        self.k = 1
        self.tree = None
        self.weights = None
        self.privateData = None
        self.norm = norm
        return

    def cost (self, simmilar, dissimilar, weights):
        l = 0.5
        simmilarCost = np.sum(Util.getWeightedNormDistanceMatrix(simmilar, weights))
        dissimilarCost =  np.sum(Util.getWeightedNormDistanceMatrix(dissimilar, weights))
        return l*simmilarCost/(len(simmilar)) - (1-l) * dissimilarCost/(len(dissimilar))

    def gradientDescent(self, simmilar, dissimilar, weights):
        #find the direction to move weights to...
        g = 0.2 #we are going to move in steps of 2.

        oldcost = self.cost(simmilar, dissimilar, weights)
        nweights = weights
        bestcost = oldcost

        for i in range(0, len(weights)):
            nweights[i] -= g
            newcost = self.cost(simmilar, dissimilar, nweights)
            
            if (newcost < bestcost):
                bestcost = newcost
            else:
                nweights[i] +=g
        return (bestcost < oldcost, nweights, bestcost)

    def getOptimalWeighting(self, simmilar, dissimilar):
        #this function accepts simmilr and dissimilar to return the optimal weightings
        #we do this by initlaizing the weight to be I and trying to find the optimal direction
        #from there that would reduce our cost function.
        weights = [1] * len(simmilar[0])
        toImprove = True
        count = 0

        while(toImprove):
            toImprove,weights,cost = self.gradientDescent(simmilar,dissimilar,weights)
            count = count + 1
            if (count == 10):
                break
        print("done weighting")
        return weights

    def getGroupData(self, data, label):
        #this is good, let's see what we can do?
        simmilar = []
        dissimilar = []
        done = False
        while(not done):
            i = random.randint(0, len(data)-1)
            j = random.randint(0, len(data)-1)
            if label[i] != label[j] and len(dissimilar) < 10000:
                dissimilar += [ Util.getNormDistanceVector(data[i], data[j], self.norm) ]
            elif label[i]  == label[j] and len(simmilar) < 10000:
                simmilar   += [ Util.getNormDistanceVector(data[i], data[j], self.norm) ]
            else:
                done = True
                break
        return (simmilar, dissimilar)

    def findKClosest(self, x):
        data  = self.privateData['X']
        label = self.privateData['Y']

        #find the k closest values.
        distances = Util.getDistanceVector(data, x, self.norm, self.weights)
        seenBefore = set();
        for i in range(0, self.k):
            min = sys.maxsize
            minindex = sys.maxsize
            for j in range(0, len(data)):
                if j in seenBefore:
                    continue;
                else:
                    if distances[j] < min:
                       min = distances[j]
                       minindex = j
            seenBefore.add(minindex)

        resultData = [data[i] for i in seenBefore]
        resultLabel = [label[i][0] for i in seenBefore]
        return (resultData, resultLabel)

    def classify(self, x):
        data, label = self.findKClosest(x)
        labelsdict = {}
        for i in label:
           if i in labelsdict:
               labelsdict[i] +=1
           else:
               labelsdict[i] = 1
        mode = 0
        index = 0
        for i in labelsdict.keys():
            if labelsdict[i] > mode:
               mode = labelsdict[i]
               index = i
        return index

    def learn(self, train):
        self.privateData = train
        simmilar, dissimilar = self.getGroupData(train['X'], train['Y'])
        self.weights = self.getOptimalWeighting(simmilar, dissimilar)
        return 0

    def run(self):
        for i in self.possibleks:
            print("KNN-k={}".format(i))
            self.k = i
            super(KNNClassifier, self).run()



class DecisionTreeClassifier(Classifier):
    def __init__(self, data, trainPercentages, runs, dimensions, treeheights):
        super(MLEClassifier, self).__init__(data, trainPercentages, runs, dimensions)
        self.dimension = 0
        self.treeheights = treeheights
        self.height = 0
        return

    def run(self):
        for i in self.treeheights:
            self.height = i
            print("Decision Tree Height = {}".format(i))
            super(DecisionTreeClassifier, self).run()




def run(datafileName, knn, pTrain, runs, dimensions):
    data = loadmat(datafileName)
    mle_classifier = MLEClassifier(data, pTrain, runs, dimensions)
    knn_classifier = KNNClassifier(data, pTrain, runs, knn, dimensions)
    mle_classifier.run()
    knn_classifier.run()
    print ("MLE Accuracy: {}, dimensions".format(mle_classifier.getAccuracy(), mle_classifier.getBestDimension() ))
    print ("{}-KNN Accuracy:{}".format(knn, knn_classifier.getAccuracy()))
    return

if __name__ == '__main__':

    orig_stdout = sys.stdout
    data  = 'hw1data.mat'
    knn = [1,2,3,4,5]
    pTrain = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    runs = 2
    dimensions =  [250]
    f = open("allout.txt", "w")
    sys.stdout =f


    run(data,knn,pTrain, runs, dimensions)
    sys.stdout = orig_stdout
    f.close()