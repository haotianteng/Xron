#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 00:04:58 2021

@author: haotian teng
"""
import os
import sys
import torch
import argparse
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from matplotlib import pyplot as plt
from xron.xron_input import Dataset, ToTensor
from xron.xron_train_base import Trainer, load_config, DeviceDataLoader
from xron.xron_model import REVCNN,DECODER_CONFIG,CRNN
from torch.distributions.one_hot_categorical import OneHotCategorical as OHC

class Evaluator(Trainer):
    def __init__(self, 
                 encoder:CRNN, 
                 decoder:REVCNN,
                 config:DECODER_CONFIG,
                 device:str = None):
        device = config.EVALUATION['device']
        super().__init__(train_dataloader=None,
                       nets = {"encoder":encoder,
                               "decoder":decoder},
                       config = config,
                       device = device,
                       eval_dataloader = None)
        self.encoder = encoder
        self.decoder = decoder
        
    def eval_once(self,batch:np.ndarray):
        encoder = self.encoder
        decoder = self.decoder
        signal = batch['signal']
        logprob = encoder.forward(signal) #[L,N,C]
        prob = torch.exp(logprob)
        m = OHC(prob)
        sampling = m.sample().permute([1,2,0]) #[L,N,C]->[N,C,L]
        rc_signal = decoder.forward(sampling).permute([0,2,1]) #[N,L,C] -> [N,C,L]
        predictions = encoder.ctc_decode(logprob,
                       alphabet = 'N' + self.config.CTC['alphabeta'],
                       beam_size = self.config.CTC['beam_size'],
                       beam_cut_threshold = self.config.CTC['beam_cut_threshold'])
        return rc_signal,prob, predictions,sampling
    
def cmd_args():
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')
    parser.add_argument('-i', '--chunks', required = True,
                        help = "The .npy file contain chunks.")
    parser.add_argument('--model_folder', required = True,
                        help = "The folder contains the trained model.")
    parser.add_argument('--seq', default = None,
                        help="The .npy file contain the sequence.")
    parser.add_argument('--seq_len', default = None,
                        help="The .npy file contain the sueqnece length.")
    parser.add_argument('--device', default = 'cuda',
                        help="The device used for training, can be cpu or cuda.")
    args = parser.parse_args(sys.argv[1:])
    return args

if __name__ == "__main__":
    args = cmd_args()
    
    #Load model
    print("Load model.")
    config = load_config(os.path.join(args.model_folder,'config.toml'))
    config.EVALUATION = {"batch_size":200,
                         "device":args.device}
    encoder = CRNN(config)
    decoder = REVCNN(config)
    e = Evaluator(encoder,decoder,config,device = args.device)
    e.load(args.model_folder)
    
    #Load data
    print("Load data.")
    chunks = np.load(args.chunks)
    reference = np.load(args.seq)
    ref_len = np.load(args.seq_len)
    dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = config.EVALUATION["batch_size"],shuffle = True, num_workers = 4)
    DEVICE = args.device
    loader = DeviceDataLoader(loader,device = DEVICE)
    
    #Evaluation
    for batch in loader:
        break
    rc_signal,prob,predictions,sampling = e.eval_once(batch)
    rc_signal = rc_signal.detach().cpu().numpy()
    prob = prob.detach().cpu().numpy()
    sampling = sampling.detach().cpu().numpy()
    norm_signal = (rc_signal - np.mean(rc_signal,axis = 2))/np.std(rc_signal,axis = 2)
    
    #Plot
    idx = np.random.randint(low = 0, high = config.EVALUATION['batch_size']-1)
    fig,axs = plt.subplots(nrows = 2,figsize = (20,20))
    axs[0].plot(norm_signal[idx,0,5:500],label = "Reconstruction")
    axs[0].plot(batch['signal'][idx,0,5:500],label = "Original signal")
    for i in np.arange(prob.shape[2]):
        axs[1].plot(prob[:,idx,i])
    axs[0].legend()
    fig.savefig(os.path.join(args.model_folder,'reconstruction.png'))
        