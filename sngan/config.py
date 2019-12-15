import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.BATCH_SIZE = 64
__C.BUFFER_SIZE = 50000
__C.EPOCHS = 20
__C.NOISE_DIM = 100
__C.NUM_EXAMPLES_TO_GENERATE = 16
__C.CHECK_DIR = './training_checkpoints'
__C.IMAGE_PATH = './saveimage'
__C.IMG_SIZE = 28
__C.SHOW_LOSS = 100
__C.DATA = 'mnist'
__C.GIF = True
__C.GAN_SN = True