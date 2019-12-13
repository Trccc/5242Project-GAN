import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import tensorflow_datasets as tfds
from pytz import timezone
from datetime import datetime
from config import cfg
import keras.backend as K



def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


def w_discriminator_loss(real_output, fake_output):
    
    real_loss = real_output
    fake_loss = fake_output
    total_loss = -tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
    return total_loss


def w_generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


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


def gp_discriminator_loss(real_output, fake_output, averaged_samples_output, averaged_samples, gradient_penalty_weight):
    loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gradient_penalty_loss(averaged_samples_output, averaged_samples, gradient_penalty_weight)
    return loss

def gp_generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)