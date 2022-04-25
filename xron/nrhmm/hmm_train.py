"""
Created on Tue Apr 19 16:19:13 2022

@author: Haotian Teng
"""
import os
import sys
import toml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from xron.nrhmm.hmm import RHMM,GaussianEmissions
from xron.nrhmm.hmm_input import Kmer2Transition, Kmer_Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from xron.xron_train_base import Trainer,DeviceDataLoader

class RHMM_Trainer(Trainer):
    def __init__(self,
                 train_dataloader:DataLoader,
                 hmm:RHMM,
                 config:object):
        super().__init__(train_dataloader,
                         nets = {"hmm":hmm},
                         config = config,
                         device = config.TRAIN["device"]
                         )
        self.hmm = hmm
        
    def train(self,epoches,optimizer,save_cycle,save_folder):
        self.save_folder = save_folder
        self._save_config()
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                loss,error = self.train_step(batch)
                if torch.isnan(loss):
                    print("NaN loss detected, skip this training step.")
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.hmm.emission.clamp_cov()
                self.losses.append(loss.item())
                self.global_step +=1
                if (i_batch+1)%save_cycle==0:
                    self.save()
                    print("Epoch %d: Batch %d, loss %f"%(epoch_i, i_batch, loss))
                    self.save_loss()

    def train_step(self,batch):
        net = self.hmm
        signal_batch = batch['signal']
        duration_batch = batch['duration']
        transition_batch = batch['labels']
        if torch.sum(torch.isnan(signal_batch)):
            print("Found NaN input signal.")
            return None,None
        log_prob = net.forward(signal_batch, duration_batch, transition_batch)
        loss = -log_prob.mean()
        return loss
    
    def train_EM(self,epoches,save_cycle,save_folder):
        self.save_folder = save_folder
        self._save_config()
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                renew = ((i_batch)%save_cycle==0)
                self.train_step_EM(batch,update_parameter = renew)
                with torch.no_grad():
                    loss = self.train_step(batch)
                self.losses.append(loss.item())
                self.global_step +=1
                if renew:
                    self.save()
                    print("Epoch %d: Batch %d, loss %f"%(epoch_i, i_batch, loss))
                    self.save_loss()

    
    def train_step_EM(self,batch,update_parameter):
        hmm = self.hmm
        signal_batch = batch['signal']
        duration_batch = batch['duration']
        transition_batch = batch['labels']
        if torch.sum(torch.isnan(signal_batch)):
            print("Found NaN input signal.")
            return None,None
        with torch.no_grad():
            gamma = hmm.expectation(signal_batch, duration_batch, transition_batch)
            hmm.maximization(signal_batch,gamma,update = update_parameter)
        return gamma.detach()
    
def train(args):
    optimizers = {'Adam':torch.optim.Adam,
                  'AdamW':torch.optim.AdamW,
                  'SGD':torch.optim.SGD,
                  'RMSprop':torch.optim.RMSprop,
                  'Adagrad':torch.optim.Adagrad,
                  'Momentum':partial(torch.optim.SGD,momentum = 0.9)}
    config = toml.load(os.path.join(args.input,"config.toml"))
    class TRAIN_CONFIG(object):
        DATA = config
        TRAIN = {"keep_record":5,
                 "grad_norm":2,
                 "inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "optimizer":optimizers[args.optimizer],
                 "device":args.device}
    train_config = TRAIN_CONFIG()
    ### Read dataset
    print("Load the dataset.")
    chunks = np.load(os.path.join(args.input,"chunks.npy"))
    durations = np.load(os.path.join(args.input,"durations.npy"))
    kmers = np.load(os.path.join(args.input,"kmers.npy"))
    k2t = Kmer2Transition(config['alphabeta'],config['k'],config['chunk_len'],config['kmer2idx_dict'],config['idx2kmer'],base_alternation = {"A":"M"}, kmer_replacement = True)
    dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
    loader = DataLoader(dataset,batch_size = args.batch_size, shuffle = True)
    loader = DeviceDataLoader(loader,device = args.device)
    
    ### Initialize the model
    print("Construct and initialize the model.")
    n_states = len(config['alphabeta'])**config['k']
    init_means = np.random.rand(n_states)-0.5 + np.mean(chunks)
    init_covs = np.asarray([np.std(chunks)*2]*n_states)
    for i in tqdm(np.arange(n_states),desc = "Initialize parameter from the given chunks."):
        selected_sig = chunks[kmers == i]
        if len(selected_sig) > 0:
            init_means[i] = np.mean(selected_sig)
    emission = GaussianEmissions(init_means[:,None], init_covs[:,None])
    hmm = RHMM(emission,normalize_transition=False)
    
    ### Train
    trainer = RHMM_Trainer(loader,hmm,config = train_config)
    # lr = args.lr
    epoches = args.epoches
    # optim = train_config.TRAIN['optimizer'](hmm.parameters(),lr = lr)
    print("Begin training the model.")
    trainer.train_EM(epoches,args.report,args.model_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training RHMM model')
    parser.add_argument("-i","--input", type = str, required = True,
                        help = "Data folder contains the chunk, kmer sequence.")
    parser.add_argument('-o', '--model_folder', required = True,
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
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.model_folder,exist_ok=True)
    train(args)
    