# NAME: Deniz Ulcay
# Image classification using different neural networks

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers


def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test
    
    y_clean = []
    yte_clean = []
    
    for item in ytrain:
        y_clean.append(item[0])
        
    for item in ytest:
        yte_clean.append(item[0])
        
    ss = tf.Session()
    ytrain_1hot = tf.one_hot(y_clean, 10)
    ytest_1hot = tf.one_hot(yte_clean, 10)
    ytrain_1hot = ytrain_1hot.eval(session = ss)
    ytest_1hot = ytest_1hot.eval(session = ss)
    
    xtrain_n = xtrain / 255
    xtest_n = xtest / 255
    
    return xtrain_n, ytrain_1hot, xtest_n, ytest_1hot


def build_multilayer_nn():
    # [loss, accuracy]
    # [1.4678429115295411, 0.4788]
    nn = Sequential()
    nn.add(Flatten(input_shape=(32,32,3)))
    
    hidden = Dense(units=100, activation="relu")
    nn.add(hidden)
    
    output = Dense(units=10, activation="softmax")
    nn.add(output)
    
    return nn


def train_multilayer_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)
    
    return None
 
    
def build_convolution_nn():
    # [loss, accuracy]
    # [0.86315463647842405, 0.70479999999999998]
    # Trained with 30 epochs    
    # [0.69824119572639465, 0.75549999999999995]
    nn = Sequential()
    
    conv1 = Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(32,32,3))
    conv2 = Conv2D(32, (3,3), activation='relu', padding="same")
    conv3 = Conv2D(32, (3,3), activation='relu', padding="same")
    conv4 = Conv2D(32, (3,3), activation='relu', padding="same")
    
    pool = MaxPooling2D(pool_size=(2,2))
    pool2 = MaxPooling2D(pool_size=(2,2))
    drop = Dropout(0.25)
    drop2 = Dropout(0.50)
    
    hidden1 = Dense(units=250, activation="relu")
    hidden2 = Dense(units=100, activation="relu")
    output = Dense(units=10, activation="softmax")

    nn.add(conv1)
    nn.add(conv2)
    nn.add(pool)
    nn.add(drop)
    nn.add(conv3)
    nn.add(conv4)
    nn.add(pool2)
    nn.add(drop2)
    nn.add(Flatten(input_shape=(8,8,32)))
    nn.add(hidden1)
    nn.add(hidden2)
    nn.add(output)
    
    return nn
    

def train_convolution_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)
    
    return None


def get_binary_cifar10():

    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test
    
    y_clean = []
    yte_clean = []
    vehicle = [0, 1, 8, 9]
    
    for item in ytrain:
        if item[0] in vehicle:
            y_clean.append(0)
        else:
            y_clean.append(1)
        
    for item in ytest:
        if item[0] in vehicle:
            yte_clean.append(0)
        else:
            yte_clean.append(1)
          
    xtrain_n = xtrain / 255
    xtest_n = xtest / 255
    
    return xtrain_n, y_clean, xtest_n, yte_clean


def build_binary_classifier():    
    # [loss, accuracy]
    # [0.15431847343444824, 0.93989999999999996]
    nn = Sequential()
    
    conv1 = Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(32,32,3))
    conv2 = Conv2D(32, (3,3), activation='relu', padding="same")
    conv3 = Conv2D(32, (3,3), activation='relu', padding="same")
    conv4 = Conv2D(32, (3,3), activation='relu', padding="same")
    
    pool = MaxPooling2D(pool_size=(2,2))
    pool2 = MaxPooling2D(pool_size=(2,2))
    drop = Dropout(0.25)
    drop2 = Dropout(0.50)
    
    hidden1 = Dense(units=250, activation="relu")
    hidden2 = Dense(units=100, activation="relu")
    output = Dense(units=1, activation="sigmoid")

    nn.add(conv1)
    nn.add(conv2)
    nn.add(pool)
    nn.add(drop)
    nn.add(conv3)
    nn.add(conv4)
    nn.add(pool2)
    nn.add(drop2)
    nn.add(Flatten(input_shape=(8,8,32)))
    nn.add(hidden1)
    nn.add(hidden2)
    nn.add(output)
    
    return nn


def train_binary_classifier(model, xtrain, ytrain):
    
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)
    
    return None




#if __name__ == "__main__":
    
    # ------ Part 1 ------
    # xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10();
    
    # ------ Part 2 ------
    # xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10();
    # nn = build_multilayer_nn();
    # train_multilayer_nn(nn, xtrain, ytrain_1hot);
    # nn.evaluate(xtest, ytest_1hot);
    
    # ------ Part 3 ------
    # xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10();
    # nn = build_convolution_nn();
    # train_convolution_nn(nn, xtrain, ytrain_1hot);
    # nn.evalutate(xtest, ytest_1hot);
    
    # ------ Part 4 ------
    # xtrain, ytrain, xtest, ytest = get_binary_cifar10();
    # nn = build_binary_classifier();
    # train_binary_classifier(nn, xtrain, ytrain);
    # nn.evalutate(xtest, ytest);
    
    
    
    
    
    
    
    
    
    
    

