
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.enable_eager_execution()

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras import backend as K

IMG_SIZE = 160
BATCH_SIZE = 64
BUFFER_SIZE = 60000
EPOCHS = 150
NOISE_DIM = 100
TRAINING_RATIO = 5  # WGAN loss parameters
GRADIENT_PENALTY_WEIGHT = 10 # As per the paper

#check if GPU is being used
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

''' Get training data '''
train_df = pd.read_csv('../train_input.csv')
training_imgs = np.load('../train_output.npz')
seq_lengths = train_df['length'].values

regions = []
for i in range(len(train_df)):
    for j in range(seq_lengths[i] // IMG_SIZE):
        regions.append(training_imgs['arr_'+str(i)][j*IMG_SIZE:(j+1)*IMG_SIZE, j*IMG_SIZE:(j+1)*IMG_SIZE])
regions = np.array(regions)
regions = np.tanh(regions.reshape((-1, IMG_SIZE, IMG_SIZE, 1))).astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices(regions).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)




# In[2]:


''' Models '''
def get_generator(img_size):
    in_size = img_size / (2**4) # Transpose Convolution naturally upsamples

    z = Input(shape = (NOISE_DIM, ))
    x = Dense(256 * in_size * in_size, use_bias=False)(z)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Reshape([in_size, in_size, 256])(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), use_bias=False, padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), use_bias=False, padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), use_bias=False, padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2DTranspose(1, (4, 4), strides=(2, 2), use_bias=False, padding='same')(x)
    y = Activation('tanh')(x) # normalize output into [-1, 1]

    return Model(z, y)

def get_discriminator(img_size):
    img = Input(shape = (img_size, img_size, 1, ))

    x = Conv2D(32, (4, 4), strides=(2, 2), padding='same')(img)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Flatten()(x)
    y = Dense(1, use_bias=False)(x)

    return Model(img, y)

generator_model = get_generator(IMG_SIZE)
generator_model.summary()

discriminator_model = get_discriminator(IMG_SIZE)
discriminator_model.summary()

''' Losses '''
def wasserstein_loss(y, y_):
    return K.mean(y * y_)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients = K.sqrt(gradients)
    gradients = gradient_penalty_weight * K.square(1 - gradients)
    return K.mean(gradients)


gen_optim = tf.train.AdamOptimizer( 0.0002, beta1=0.5, beta2=0.9)
dis_optim = tf.train.AdamOptimizer( 0.0002, beta1=0.5, beta2=0.9)

generator_model.compile(optimizer=gen_optim, loss=wasserstein_loss)
discriminator_model.compile(optimizer=dis_optim, loss=wasserstein_loss)


'''
Setup checkpoints in case of failure
Provide function for storing generated images
'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optim,
                                 discriminator_optimizer=dis_optim,
                                 generator=generator_model,
                                 discriminator=discriminator_model)

num_examples_to_generate = 16
random_vector_for_generation = tf.random_normal([num_examples_to_generate, NOISE_DIM])
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    plt.savefig('./gan_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

''' Training

    This training was written back when the vanilla losses were still in this
    file. This is no longer the case. Essentially this needs to be updated to
    use train_on_batches. Models need to be properly connected and WGAN loss
    used to properly update.
    See improved WGAN paper:
        https://arxiv.org/pdf/1704.00028.pdf
    See a Keras implementation of improved WGAN loss:
        https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
'''


# In[37]:


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        
def train(dataset, epochs, noise_dim):
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    for epoch in range(epochs):
        start = time.time()

        print("Epoch: ", epoch)
        print("Number of batches: ", int(dataset.shape[0] // BATCH_SIZE))
        print("Dataset.shape[0]: ", dataset.shape[0])

        discriminator_loss = []
        generator_loss = []

        minibatches_size = BATCH_SIZE * TRAINING_RATIO

        for i in range(int(dataset.shape[0]) // (BATCH_SIZE * TRAINING_RATIO)):
            discriminator_minibatches = dataset[i * minibatches_size:(i + 1) * minibatches_size]
            print("discriminator_minibatches.shape: ", discriminator_minibatches.shape)
            for j in range (TRAINING_RATIO):
                ## Generator Images ##
                image_batch = discriminator_minibatches[j * BATCH_SIZE : (j+1) * BATCH_SIZE]
                noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
                generated_images = generator_model.predict(noise_gen)
            
                X = np.concatenate((image_batch, generated_images))
                #print("X.shape: ", X.shape)
                y = np.zeros([2*BATCH_SIZE,2])
                y[0:BATCH_SIZE,1] = 1
                y[BATCH_SIZE:,0] = 0
                
                ## Train Discriminator ##
                make_trainable(discriminator_model,True)
                discriminator_loss.append(discriminator_model.train_on_batch(X , y))
                
                noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
                y2 = np.ones([BATCH_SIZE,160,160,1])
                y2[:,1] = 1
                #print(y2)
                
                ## Train Generator ##
                make_trainable(discriminator_model,True)
                generator_loss.append(generator_model.train_on_batch(noise_tr, y2))
        
        generate_and_save_images(generator_model, epoch + 1, random_vector_for_generation)
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoc {} is {} sec'.format(epoch + 1, time.time()-start))
    generator_model.save('gen.h5')
    discriminator_model.save('dis.h5')

train(regions, EPOCHS, NOISE_DIM)

