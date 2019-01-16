import os
import time
import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Layer, Input, Dense, Reshape, Flatten, ReLU, Dropout, Permute
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
import tensorflow as tf
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOISE_DIM = 100
IMG_SIZE = 160
BATCH_SIZE = 64
EPOCHS = 250
TRAINING_RATIO = 5  # WGAN loss parameters
GRADIENT_PENALTY_WEIGHT = 10 # As per the paper
# random_vector_for_generation = np.random.rand(16, NOISE_DIM).astype(np.float32)
random_vector_for_generation = np.random.normal(size=(16, NOISE_DIM))

''' Models '''
# used as final layer of generator to ensure symmetric matrices
class Symmetrize(Layer):
    def __init__(self, **kwargs):
        super(Symmetrize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Symmetrize, self).build(input_shape)

    def call(self, x):
        x_ = Permute([2, 1, 3])(x)
        x = (x+x_)/2
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

def get_generator(img_size):
    in_size = img_size / (2**5) # Transpose Convolution naturally upsamples
    model = Sequential()

    model.add(Dense(256 * in_size * in_size, use_bias=False, input_dim=NOISE_DIM))
    model.add(LeakyReLU(0.2))
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Reshape((in_size, in_size, 256), input_shape=(256 * in_size * in_size, )))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), use_bias=False, padding='same'))
    model.add(LeakyReLU(0.2))
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), use_bias=False, padding='same'))
    model.add(LeakyReLU(0.2))
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), use_bias=False, padding='same'))
    model.add(LeakyReLU(0.2))
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2DTranspose(1, (4, 4), strides=(2, 2), use_bias=False, padding='same'))
    model.add(LeakyReLU(0.2))
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='tanh', use_bias=False, padding='same'))
    model.add(Symmetrize())

    return model

def get_discriminator(img_size):
    model = Sequential()

    model.add(Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1, use_bias=False))

    return model

''' Losses '''
def wasserstein_loss(y, y_):
    return K.mean(y * y_)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients = K.square(gradients)
    gradients = K.sum(gradients, axis=np.arange(1, len(gradients.shape)))
    gradients = K.sqrt(gradients)
    gradients = gradient_penalty_weight * K.square(1 - gradients)
    return K.mean(gradients)

''' Get point on line between real and generated input '''
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

''' Get training data '''
train_df = pd.read_csv('../train_input.csv')
training_imgs = np.load('../train_output.npz')
seq_lengths = train_df['length'].values

# break up training data into non-overlapping portions of the desired size
train_dataset = []
for i in range(len(train_df)):
    for j in range(seq_lengths[i] / IMG_SIZE):
        train_dataset.append(training_imgs['arr_'+str(i)][j*IMG_SIZE:(j+1)*IMG_SIZE, j*IMG_SIZE:(j+1)*IMG_SIZE])
train_dataset = np.array(train_dataset)
max = train_dataset.max((1,2)).reshape((train_dataset.shape[0],1,1))
train_dataset = 2 * train_dataset / max - 1 # normalize between -1, 1
train_dataset = train_dataset.reshape((-1, IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)

''' Create the models '''
generator = get_generator(IMG_SIZE)
discriminator = get_discriminator(IMG_SIZE)

''' Model compilation following
https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py '''
''' Prepearing generator for training '''
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(NOISE_DIM,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
generator_model.compile(optimizer=Adam(0.0002, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

''' Preparing discriminator for training '''
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

real_samples = Input(shape=train_dataset.shape[1:])
generator_input_for_discriminator = Input(shape=(NOISE_DIM,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# Weighted average of real and generated samples
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
averaged_samples_out = discriminator(averaged_samples)

# Setting up gradient loss term
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
discriminator_model.compile(optimizer=Adam(0.0002, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])

''' Summarize both models '''
generator_model.summary()
discriminator_model.summary()

''' Setup generation and saving of images '''
def generate_images(model, epoch):
    predictions = model.predict(random_vector_for_generation)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    plt.savefig('./gan_images7/image_at_epoch_{:04d}.png'.format(epoch+1))
    plt.close()

''' Training '''
def train(dataset, epochs, noise_dim):
    # ground truth
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
    # positive_y = np.random.uniform(0.8, 1.2, (BATCH_SIZE, 1))
    # negative_y = -np.random.uniform(0.8, 1.2, (BATCH_SIZE, 1))

    for epoch in range(epochs):
        start = time.time()
        X_train = dataset
        np.random.shuffle(X_train)
        print("Epoch: " + str(epoch+1))
        print("Number of batches: " + str(int(X_train.shape[0] // BATCH_SIZE)))
        discriminator_loss = []
        generator_loss = []
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
            discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                # noise = np.random.rand(BATCH_SIZE, NOISE_DIM).astype(np.float32)
                noise = np.random.normal(size=(BATCH_SIZE, NOISE_DIM)).astype(np.float32)
                discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise], [positive_y, negative_y, dummy_y]))
            # noise = np.random.rand(BATCH_SIZE, NOISE_DIM).astype(np.float32)
            noise = np.random.normal(size=(BATCH_SIZE, NOISE_DIM)).astype(np.float32)
            generator_loss.append(generator_model.train_on_batch(noise, positive_y))

        generate_images(generator, epoch)
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        if (epoch+1) % 15 == 0:
            generator.save('./training_checkpoints/gen'+str(epoch+1)+'.h5')
            discriminator.save('./training_checkpoints/dis'+str(epoch+1)+'.h5')

    generator.save('gen.h5')
    discriminator.save('dis.h5')

train(train_dataset, EPOCHS, NOISE_DIM)
