import tensorflow as tf
from tensorflow.keras import layers


def make_generator_model_mnist():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,), activation='relu'))
    model.add(layers.Dense(512, use_bias=False, activation='relu'))
    model.add(layers.Dense(1152, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((3, 3, 128)))
    assert model.output_shape == (None, 3, 3, 128) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model_mnist():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model



def make_generator_model_svhn():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, use_bias=False, input_shape=(100,), activation='relu'))
    model.add(layers.Dense(512, use_bias=False, activation='relu'))
    model.add(layers.Dense(4096, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((4, 4, 256)))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), use_bias=False))
    model.add(layers.Cropping2D(1))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), use_bias=False))
    model.add(layers.Cropping2D(1))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), use_bias=False, activation='tanh'))
    model.add(layers.Cropping2D(1))
    assert model.output_shape == (None, 32, 32, 3)
    return model

def make_discriminator_model_svhn_p():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), padding='same', use_bias=False, input_shape=[32, 32, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size = (2,2)))
    model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(1, 1),padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size = (2,2)))
    model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (4, 4), strides=(1, 1),use_bias=False,padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size = (2,2)))
    model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Dropout(0.3))   

    model.add(layers.Flatten())
    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(64, use_bias=False))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(16, use_bias=False))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(1))

    return model

def make_discriminator_model_svhn():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
#     model.add(layers.Dropout(0.3))   

    model.add(layers.Flatten())
    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(64, use_bias=False))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(16, use_bias=False))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(1))

    return model