import tensorflow as tf
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


def generate_and_save_images(model, epoch, test_input,savedir):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        if cfg.DATA == 'mnist':
            plt.imshow(np.uint8(predictions[i, :, :, 0] * 127.5 + 127.5), cmap='gray')
        elif cfg.DATA.lower() == 'svhn':
            plt.imshow(np.uint8(predictions[i, :, :, :] * 127.5 + 127.5))
        plt.axis('off')
    path = os.path.join(savedir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(path)
    plt.close('all')

    
@tf.function
def format_example(data):
    # First, convert the data type to tf.float32
    image = data["image"]
#     label = data["label"]
    image =  tf.cast(image, tf.float32)
    # Second, normalize the image
    image = (image - 127.5) / 127.5
    # Third, resize the image 
    image = tf.image.resize(image,[cfg.IMG_SIZE,cfg.IMG_SIZE])
    return image

def get_train_data(name = 'svhn'):
    
    if name.lower() == 'svhn':
        
        (train_data,_), info = tfds.load(name="svhn_cropped", split=["train","test"], shuffle_files=True, with_info=True)

        BUFFER_SIZE = cfg.BUFFER_SIZE
        BATCH_SIZE = cfg.BATCH_SIZE

        return train_data.map(format_example).shuffle(cfg.BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    elif name.lower() =='mnist':
        
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], cfg.IMG_SIZE, cfg.IMG_SIZE, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5
        train_images = tf.data.Dataset.from_tensor_slices(train_images).shuffle(cfg.BUFFER_SIZE).batch(cfg.BATCH_SIZE)
        return train_images