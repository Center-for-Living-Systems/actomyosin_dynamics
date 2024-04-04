# a wrapper of AE running
# Liya Ding
# 2024.03

import tensorflow as tf
from ../utils import actomyosin_data, plot64
from ../model.autoencoder_64 import AE, VAE, CVAE
from ../train_utils.autoencoder import AETrain, VAETrain, CVAETrain
import time
import argparse
from datetime import datetime
from packaging import version
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import tifffile
import os
import warnings
from itertools import cycle, islice

from skimage.transform import resize
from skimage.io import imsave, imread

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler



def parse_args():
    desc = "Tensorflow 2.0 implementation of 'AutoEncoder Families (AE, VAE, CVAE(Conditional VAE))'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ae_type', type=str, default=False,
                        help='Type of autoencoder: [AE, VAE, CVAE]')
    parser.add_argument('--net_type', type=str, default=False,
                        help='Type of ae layers: [simple, conv]')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Degree of latent dimension(a.k.a. "z")')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='The number of training epochs')
    parser.add_argument('--learn_rate', type=float, default=1e-4,
                        help='Learning rate during training')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--train_buf', type=int, default=1000,
                        help='Train buf')    
    return parser.parse_args()


def main(args):    
    ae_type = args.ae_type
    net_type = args.net_type
    latent_dim = args.latent_dim,
    num_epochs = args.num_epochs,
    learn_rate = args.learn_rate,
    batch_size = args.batch_size
    train_buf = args.train_buf 

    

if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()
    main(args)
