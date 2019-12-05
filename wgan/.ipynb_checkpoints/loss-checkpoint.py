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
    
    real_loss = real_output
    fake_loss = fake_output
    total_loss = -tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
    return total_loss


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)