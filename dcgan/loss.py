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



def discriminator_loss(real_output, fake_output):
    # label flip
    if tf.random.uniform([1]) < 0.05:
        real_output, fake_output = fake_output, real_output
        
    # with label smoothing
    real_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1, from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1, from_logits=True)(tf.zeros_like(fake_output), fake_output)
    
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