import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import IPython
import tensorflow_datasets as tfds
from pytz import timezone
from datetime import datetime
from config import cfg
from model import *
from utils import *
from loss import *
from Inception_score import *


@tf.function
def train_step(images,showloss = False):
    noise = tf.random.normal([cfg.BATCH_SIZE, cfg.NOISE_DIM])
    
    g_loss = generator_loss
    d_loss = discriminator_loss
        
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = g_loss(fake_output)
        disc_loss = d_loss(real_output, fake_output)
        
#         if showloss:
#             print('gen_loss = %.4f|disc_loss = %.4f'%(gen_loss.numpy(),disc_loss.numpy()))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
        
    
def train(dataset, epochs, savedir):
    IS_mean = []
    IS_std = []
    for epoch in range(epochs):
        start = time.time()
        i = 0
        for image_batch in dataset:
            i += 1
            if (i+1) % cfg.SHOW_LOSS ==0:
                train_step(image_batch,showloss = True)
            else:
                train_step(image_batch)
        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed,savedir)
        
        # Save the model every 15 epochs
        if (epoch + 1) % 5 == 0:
            mean, std = IS(generator, 1000, 100)
            IS_mean.append(mean)
            IS_std.append(std)
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    # clear outputs
    display.clear_output(wait=True)
    
    # save IS score and plot
    IS_mean = np.array(IS_mean)
    IS_std = np.array(IS_std)
    IS_df = pd.DataFrame({'mean':IS_mean, 'mean+std':IS_mean+IS_std, 'mean-std':IS_mean-IS_std, 'std':IS_std})
    df_path = os.path.join(savedir, 'IS_score.csv')
    IS_df.to_csv(path_or_buf=df_path, index=False)
    print('Inception score save complete')
    path = os.path.join(savedir, 'IS_score_trend.png')
    fig = plt.figure(figsize=(4,4))
    plt.plot(IS_df[['mean','mean+std','mean-std']])
    plt.legend(IS_df[['mean','mean+std','mean-std']].columns, loc='best')
    plt.savefig(path)
    
    # Generate after the final epoch
    generate_and_save_images(generator,
                           epochs,
                           seed,savedir)


if __name__ == '__main__':
    
    if cfg.DATA == 'mnist':

        train_data = get_train_data('mnist')
        generator = make_generator_model_mnist()
        discriminator = make_discriminator_model_mnist()
        
    elif cfg.DATA == 'svhn':
        
        train_data = get_train_data('svhn')
        generator = make_generator_model_svhn()
        discriminator = make_discriminator_model_svhn_p()
    
    noise = tf.random.normal([1, 100])
    
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, beta_1 = 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate =1e-4, beta_1 = 0.5)

    
#     checkpoint_dir = cfg.CHECK_DIR
#     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#     checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                      discriminator_optimizer=discriminator_optimizer,
#                                      generator=generator,
#                                      discriminator=discriminator)
    
    EPOCHS = cfg.EPOCHS
    noise_dim = cfg.NOISE_DIM
    num_examples_to_generate = cfg.NUM_EXAMPLES_TO_GENERATE
    
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    now = datetime.now(timezone('US/Eastern'))
    
    subfile = now.strftime("%m_%d_%H_%M")
    
    filedir = os.path.join(cfg.IMAGE_PATH,subfile)
    
    checkpoint_dir = filedir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    if not os.path.exists(cfg.IMAGE_PATH):
        os.mkdir(cfg.IMAGE_PATH)
   
    if not os.path.isfile(filedir):
        os.mkdir(filedir)
                           
    savedir = filedir
          
    train(train_data, EPOCHS,savedir)
    
    if cfg.GIF:
        anim_file = subfile+'gan.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(filedir+'/image*.png')
            filenames = sorted(filenames)
            last = -1
            for i,filename in enumerate(filenames):
                frame = 2*(i**0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
    print('finish')