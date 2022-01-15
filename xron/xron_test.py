#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 00:04:58 2021

@author: haotian teng
"""
import os
import sys
import umap
import torch
import argparse
import numpy as np
import torch.utils.data as data
from itertools import islice
from typing import List,Union
from torchvision import transforms
from matplotlib import pyplot as plt
from xron.xron_input import Dataset, ToTensor
from xron.xron_train_base import Trainer, load_config, DeviceDataLoader
from xron.xron_model import REVCNN,DECODER_CONFIG,CRNN,MM
from torch.distributions.one_hot_categorical import OneHotCategorical as OHC

class Evaluator(Trainer):
    def __init__(self, 
                 encoder:CRNN, 
                 decoders:List[Union[REVCNN,MM]],
                 config:DECODER_CONFIG,
                 device:str = None):
        device = config.EVALUATION['device']
        super().__init__(train_dataloader=None,
                       nets = {"encoder":encoder,
                               "decoder":decoders[0],
                               "mm":decoders[1]},
                       config = config,
                       device = device,
                       eval_dataloader = None)
        self.encoder = encoder
        self.decoder_revcnn, self.decoder_mm = decoders
        
    def eval_once(self,batch:np.ndarray):
        encoder = self.encoder
        d1 = self.decoder_revcnn
        d2 = self.decoder_mm
        signal = batch['signal']
        logprob = encoder.forward(signal) #[L,N,C]
        prob = torch.exp(logprob)
        m = OHC(prob)
        sampling = m.sample().permute([1,2,0]) #[L,N,C]->[N,C,L]
        rc_signal = d1.forward(sampling).permute([0,2,1]) #[N,L,C] -> [N,C,L]
        if d2:
            rc_signal += d2.forward(sampling,device = self.device).permute([0,2,1])
        predictions = encoder.ctc_decode(logprob,
                       alphabet = 'N' + self.config.CTC['alphabeta'],
                       beam_size = self.config.CTC['beam_size'],
                       beam_cut_threshold = self.config.CTC['beam_cut_threshold'])
        return rc_signal,prob, predictions,sampling

class VQ_Evaluator(Trainer):
    def __init__(self, 
                 encoder:CRNN, 
                 decoders:List[Union[REVCNN,MM]],
                 config:DECODER_CONFIG,
                 device:str = None):
        device = config.EVALUATION['device']
        super().__init__(train_dataloader=None,
                       nets = {"encoder":encoder,
                               "decoder":decoders[0],
                               "mm":decoders[1]},
                       config = config,
                       device = device,
                       eval_dataloader = None)
        self.encoder = encoder
        self.decoder_revcnn, self.decoder_mm = decoders
        self.umap_transformer = umap.UMAP()
        
    def eval_once(self,batch:np.ndarray,umap_visualize = True):
        encoder = self.encoder
        d1 = self.decoder_revcnn
        d2 = self.decoder_mm
        # embedding = d2.level_embedding
        signal = batch['signal']
        embed = encoder.forward_wo_fnn(signal) #[L,N,C]
        if umap_visualize:
            u = self.umap_embedding(embed)
        rc_signal = d1.forward(embed).permute([0,2,1]) #[N,L,C] -> [N,C,L]
        return rc_signal,u
    
    def umap_embedding(self,embedding:np.ndarray):
        u = self.umap_transformer.fit_transform(embedding.view(-1,embedding.shape[-1]).detach().cpu().numpy())
        return u
    

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
    parser.add_argument('--repeat', type = int, default = 5,
                        help="The repeat used to test.")
    parser.add_argument('--method',default = "VQ",
                        help="The embedding method used to train, can be VQ or MM")
    args = parser.parse_args(sys.argv[1:])
    return args

if __name__ == "__main__":
    args = cmd_args()
    
    #Load model
    print("Load model.")
    config = load_config(os.path.join(args.model_folder,'config.toml'))
    stride = config.CNN['Layers'][-1]['stride']
    config.EVALUATION = {"batch_size":100,
                         "device":args.device}
    encoder = CRNN(config)
    revcnn = REVCNN(config) if 'CNN_DECODER' in config.__dict__.keys() else None
    mm = MM(config) if 'PORE_MODEL' in config.__dict__.keys() else None
    if args.method == "VQ":
        e = VQ_Evaluator(encoder,[revcnn,mm],config,device = args.device)
    elif args.method == "MM":
        e = Evaluator(encoder,[revcnn,mm],config,device = args.device)
    e.load(args.model_folder)
    
    #Load data
    print("Load data.")
    chunks = np.load(args.chunks)
    if args.seq:
        reference = np.load(args.seq)
    else:
        reference = None
    if args.seq_len:
        ref_len = np.load(args.seq_len)
    else:
        ref_len = None
    dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = config.EVALUATION["batch_size"],shuffle = True, num_workers = 4)
    DEVICE = args.device
    loader = DeviceDataLoader(loader,device = DEVICE)
    
    #Evaluation
    batch = next(islice(loader,2,None))
    rc_signal, umap_vis = e.eval_once(batch)
    rc_signal = rc_signal.detach().cpu().numpy()
    norm_signal = (rc_signal - np.mean(rc_signal,axis = 2))/np.std(rc_signal,axis = 2)
    
    #Plot
    for r in np.arange(args.repeat):
        idx = np.random.randint(low = 0, high = config.EVALUATION['batch_size']-1)
        fig,axs = plt.subplots(nrows = 2,figsize = (20,30),gridspec_kw={'height_ratios': [1, 2]})
        start_idx =0
        last_idx = 800
        axs[0].plot(norm_signal[idx,0,start_idx:last_idx],label = "Reconstruction")
        axs[0].plot(batch['signal'].cpu()[idx,0,start_idx:last_idx],label = "Original signal")
        axs[0].legend()
        axs[1].scatter(umap_vis[:,0],umap_vis[:,1])
        fig.savefig(os.path.join(args.model_folder,'reconstruction_%d.png'%(r)))
        