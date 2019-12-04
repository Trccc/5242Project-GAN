import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.BATCH_SIZE = 256
__C.BUFFER_SIZE = 60000
__C.EPOCHS = 10
__C.NOISE_DIM = 100
__C.NUM_EXAMPLES_TO_GENERATE = 16
__C.CHECK_DIR = './training_checkpoints'
__C.IMAGE_PATH = './saveimage/wgan_clip'
__C.IMG_SIZE = 28
__C.WGAN_CLIP = False
__C.CLIP = 0.01
__C.SHOW_LOSS = 100
__C.DATA = 'mnist'