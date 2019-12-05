import numpy as np
import scipy.io as sio
import math
from skimage.transform import resize
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)

# input images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
    model = InceptionV3()
    scores = list()
    n_part = math.floor(images.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        subset = scale_images(subset, (299, 299, 3))
        subset = subset.astype('float32')
        subset = preprocess_input(subset)
        p_yx = model.predict(subset)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

def IS(model, n_sample, noise_dim): 
    # n_sample=1000 will take less than 1 minute
    new_seed = tf.random.normal([n_sample, noise_dim])
    predictions = model(new_seed, training=False)
    predictions = (predictions + 1) / 2 * 255
    is_avg, is_std = calculate_inception_score(predictions)
    print('score is {} ± {}'.format(is_avg, is_std))