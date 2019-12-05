import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
from config import cfg
from model import *
from inception_score import *
from keras.layers.merge import _Merge
import keras.backend as K

# load SVHN data
train_location = 'train_32x32.mat'

# read SVHN train data from local file
def load_train_data(location):
    train_data = sio.loadmat(location)
    X_train, y_train = train_data['X'], train_data['y']

    X_train = np.rollaxis(X_train, 3)

    for i in range(len(y_train)):
        if y_train[i]%10 == 0:
            y_train[i] = 0

    return (X_train, y_train)

    
def gradient_penalty_loss(averaged_samples_output, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(averaged_samples_output, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def discriminator_loss(real_output, fake_output, averaged_samples_output, averaged_samples, gradient_penalty_weight):
    loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gradient_penalty_loss(averaged_samples_output, averaged_samples, gradient_penalty_weight)
    return loss

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        weights = K.random_uniform([BATCH_SIZE, 1, 1, 1])
        averaged_samples = generated_images + weights * (images - generated_images)
        averaged_samples_output = discriminator(averaged_samples, training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, averaged_samples_output, averaged_samples, gradient_penalty_weight)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        # Produce images for the GIF as we go
#         display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
##            IS(generator, 1000, 100)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    #os.mkdir(cfg.IMAGE_PATH)
    path = os.path.join(cfg.IMAGE_PATH, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(path)
#     plt.show()

if __name__ == '__main__':

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    
    
    BUFFER_SIZE = cfg.BUFFER_SIZE
    BATCH_SIZE = cfg.BATCH_SIZE
    gradient_penalty_weight = cfg.GRADIENT_PENALTY_WEIGHT
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    generator = make_generator_model()
   
    discriminator = make_discriminator_model()
##    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    
    
##    generator_optimizer = tf.keras.optimizers.RMSprop(0.00005)
##    discriminator_optimizer = tf.keras.optimizers.RMSprop(0.00005)
    
    checkpoint_dir = cfg.CHECK_DIR
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    EPOCHS = cfg.EPOCHS
    noise_dim = cfg.NOISE_DIM
    num_examples_to_generate = cfg.NUM_EXAMPLES_TO_GENERATE
    
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    train(train_dataset, EPOCHS)
