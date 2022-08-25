"""
Created on Mon May  9 06:15:29 2022

@author: Haotian Teng
"""
import os
import sys
import time
import toml
import torch
import argparse
import itertools
import numpy as np
from matplotlib import pyplot as plt
from xron.nrhmm.hmm import GaussianEmissions, RHMM
from xron.nrhmm.hmm_input import Kmer2Transition, Kmer_Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from xron.xron_train_base import DeviceDataLoader





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluating using the RHMM model')
    parser.add_argument("-i","--input", type = str, required = True,
                        help = "Data folder contains the chunk, kmer sequence.")
    parser.add_argument('-o', '--output', required = True,
                        help = "The folder to save folder at.")
    parser.add_argument("-b","--batch_size", type = int, default = 50,
                        help = "The batch size to train.")
    parser.add_argument("--lr",type = float, default = 1e-2,
                        help = "The initial training learning rate.")
    parser.add_argument("--report",type = int, default = 10,
                        help = "Report the loss and save the model every report cycle.")
    parser.add_argument("--certain_methylation",action = "store_false", 
                        dest = "kmer_replacement", 
                        help = "If we are sure about the methylation state.")
    parser.add_argument("--optimizer",type = str, default = "Adam",
                        help = "The optimizer used to train the model.")
    parser.add_argument('--epoches', default = 10, type = int,
                        help = "The number of epoches to train.")
    parser.add_argument("--device",type = str, default = "cuda",
                        help = "The device used to train the model.")
    parser.add_argument('--load', dest='retrain', action='store_true',
                        help='Load existed model.')
    parser.add_argument('--moving_average', type = float, default = 0.0,
                         help="The factor of moving average, 0 means no delay.")
    parser.add_argument('--trainable_bases', type = str, default = None,
                        help="A magic string AB!CD or AB that gives the trainable bases and NOT trainalbe bases, for example MC means trains on kmer that must contains M and C, A!M means trains on kmer that must contains A but not contains M.")
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.model_folder,exist_ok=True)
    train(args)
    