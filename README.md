# 5242Project

Final Project for 5242 Advanced Machine Learning 

## Members

Chirong Zhang cz2533

Zhichao Liu zl2686

Yunxiao Zhao yz3380

Yusang Mao ym2694

## Tasks

report

• From the paper. https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf, understand core ideas of GAN. Make sure to understand Figure 1 and Algorithm 1 of the paper. Also, why we want G to minimize and D to maximize V (G, D).

• Implement your own GAN with CNN layers on MNIST data. Please describe the architectures of your generator and discriminator and also any hyper-parameters chosen. Post a plot of training process from tensorboard to make sure that the networks are trained as expected. To help guide you, an example of GAN on MNIST can be found in https://www.tensorflow.org/tutorials/generative/dcgan, but importantly, you must develop your own code and your own neural network models.

• Visualize samples from your model. How do they look compared to real data? How is it compared to Figure 2a in the paper?

• Implement your own GAN with SVHN data. Explore diﬀerent architecture of neural networks and hyperparameters. Compare samples from your model to real data. How is the quality compared to your GAN on MNIST? If the training does not go well, what failure modes do you see?

• (Optional) There are several improved versions of GAN such as Wasserstein GAN (WGAN). Train your own WGAN https://arxiv.org/abs/1701.07875 on MNIST and SVHN instead of the plain GAN.


Inception score -- 李宏毅 [Lecture 10](https://www.youtube.com/watch?v=IB_ADssBomk&list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw&index=10)  
[Keras Implementation](https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/)

Change CNN architecture.

Try original GAN on SVHN data

reference[blog](https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/?nsukey=LKALNIt1JkY2XrdT3fIBlKyQGMaD93R%2BvZofl8M9SJY4JnDH%2FZ3%2FdeZMlbVlh%2ByoJ1QBGzsG5rKTKxul4Rf7pG7Pbe2yzuCQbiRym%2FHAZN8aBc4WWOOcmGwQwYHAHFyWeLfq4%2B%2FaaEyVrzKIXtrxwHggMcT0hwrEx4jHLE014qX0pxO%2FI%2Fc9umB%2Fy4j1JuMjVYswlw8%2FrpJKCxJWClp7Tg%3D%3D)

Implement WGAN[Lecture 6](https://www.youtube.com/watch?v=3JP-xuBJsyc&list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw&index=6)
- clip [Keras Implementation](https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py)   
- gradient penalty wgan-gp [Keras Implementation](https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py)  
Spectrual Normalization [Keras Implementation](https://github.com/IShengFang/SpectralNormalizationKeras)

## Challenge encountered

momentum 0.5


## reference
