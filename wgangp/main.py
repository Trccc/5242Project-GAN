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
from Inception_score import *
from utils import *
from loss import *
import pandas as pd



@tf.function
def train_step(images, showloss = False):
    noise = tf.random.normal([cfg.BATCH_SIZE, cfg.NOISE_DIM])
    
    g_loss = gp_generator_loss
    d_loss = gp_discriminator_loss
        
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        weights = K.random_uniform([cfg.BATCH_SIZE, 1, 1, 1])
        averaged_samples = generated_images + weights * (images - generated_images)
        averaged_samples_output = discriminator(averaged_samples, training=True)

        gen_loss = g_loss(fake_output)
        disc_loss = d_loss(real_output, fake_output, averaged_samples_output, averaged_samples, cfg.GRADIENT_PENALTY_WEIGHT)
        
        #if showloss:
            #print('gen_loss = %.4f|disc_loss = %.4f'%(gen_loss.numpy(),disc_loss.numpy()))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss
    
def train(dataset, epochs, savedir):
    IS_mean = []
    IS_std = []
    G_loss = []
    D_loss = []
    for epoch in range(epochs):
        start = time.time()
        i = 0
        g_loss = 0
        d_loss = 0
        for image_batch in dataset:
            i += 1
            if (i+1) % cfg.SHOW_LOSS ==0:
                g_tensor, d_tensor = train_step(image_batch, showloss = True)
            else:
                g_tensor, d_tensor = train_step(image_batch)
            g_loss += float(g_tensor.numpy())
            d_loss += float(d_tensor.numpy())
        
        G_loss.append(g_loss / i)
        D_loss.append(d_loss / i)
        # Produce images for the GIF
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed,savedir)
        
        if (epoch + 1) % 5 == 0:
            mean, std = IS(generator, 1000, 100)
            IS_mean.append(mean)
            IS_std.append(std)
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    # clear outputs
    display.clear_output(wait=True)
    
    # save IS score and Loss plot
    IS_mean = np.array(IS_mean)
    IS_std = np.array(IS_std)
    IS_df = pd.DataFrame({'mean':IS_mean, 'mean+std':IS_mean+IS_std, 'mean-std':IS_mean-IS_std, 'std':IS_std})
    IS_df.index = [5 * (x + 1) for x in range(IS_df.shape[0])]
    Loss_df = pd.DataFrame({'Generater':G_loss, 'Discriminator':D_loss})
    
    df_path = os.path.join(savedir, 'IS_score.csv')
    IS_df.to_csv(path_or_buf=df_path, index=False)
    df_path2 = os.path.join(savedir, 'Loss.csv')
    Loss_df.to_csv(path_or_buf=df_path2, index=False)
    print('Inception score and loss save complete')
    
    path = os.path.join(savedir, 'IS_score_trend.png')
    fig = plt.figure(figsize=(6, 6))
    plt.plot(IS_df[['mean','mean+std','mean-std']])
    plt.title('Inception Scores')
    plt.legend(IS_df[['mean','mean+std','mean-std']].columns, loc='best')
    plt.savefig(path)
    #plt.close('all')
    
    path2 = os.path.join(savedir, 'Loss_trend.png')
    fig2 = plt.figure(figsize=(6, 6))
    plt.plot(Loss_df)
    plt.title('Validation Losses')
    plt.legend(Loss_df.columns, loc='best')
    plt.savefig(path2)
    
    # Generate after the final epoch
    generate_and_save_images(generator,
                           epochs,
                           seed,savedir)
    
    


if __name__ == '__main__':
    
    if cfg.DATA.lower() == 'mnist':

        train_data = get_train_data('mnist')
        generator = make_generator_model_mnist()
        discriminator = make_discriminator_model_mnist()
        
    elif cfg.DATA.lower() == 'svhn':
        
        train_data = get_train_data('svhn')
        generator = make_generator_model_svhn()
        discriminator = make_discriminator_model_svhn_p_no_bn()
    
    noise = tf.random.normal([1, 100])
    
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9)

    
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
          
    train(train_data, EPOCHS, savedir)
    
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
