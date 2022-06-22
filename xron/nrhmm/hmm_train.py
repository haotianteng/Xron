"""
Created on Tue Apr 19 16:19:13 2022

@author: Haotian Teng
"""
import os
import sys
import time
import toml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from functools import partial
from xron.nrhmm.hmm import RHMM,GaussianEmissions
from xron.nrhmm.hmm_input import Kmer2Transition, Kmer_Dataset, Normalizer
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
            if epoch_i > 0:
                max_variance = np.max(self.hmm.emission.certainty.cpu().numpy())
                print("Rescale the covariance to %.2f"%(max_variance))
                self.hmm.emission.cov[:] = self.config.TRAIN["exploration"]*max_variance
            for i_batch, batch in enumerate(self.train_ds):
                start = time.time()
                loss = self.train_step(batch)
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
                    time_elapsed = time.time() - start
                    self.save()
                    print("Epoch %d: Batch %d, loss %f, time elapsed per batch %.2f"%(epoch_i, i_batch, loss, time_elapsed))
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
        loss = -torch.logsumexp(log_prob, dim = 1).mean()
        return loss
    
    def train_EM(self,epoches,save_cycle,save_folder):
        self.save_folder = save_folder
        self._save_config()
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                start = time.time()
                renew = ((i_batch)%save_cycle==0)
                self.train_step_EM(batch,update_parameter = renew)
                self.global_step +=1
                if renew:
                    time_elapsed = time.time() - start
                    with torch.no_grad():
                        loss = self.train_step(batch)
                    self.losses.append(loss.item())
                    self.save()
                    print("Epoch %d: Batch %d, loss %f, time elapsed per batch %.2f"%(epoch_i, i_batch, loss, time_elapsed))
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
            hmm.maximization(signal_batch,
                             gamma,
                             update = update_parameter,
                             lr=self.config.TRAIN["learning_rate"],
                             weight_decay = self.config.TRAIN["weight_decay"],
                             momentum = self.config.TRAIN["momentum"])
        return gamma.detach()

def trainable_kmer(idx2kmer:List,must_contain:str,exclude_bases:str)->np.array:
    """
    Generate the trainble mask given the trainable bases

    Parameters
    ----------
    idx2kmer : List
        The index to kmer map.
    must_contain : str, optional
        Bases that must exist in the kmer to make the kmer trainable.
    exlcude_bases : str, optional
        Bases that should NOT exist in the kmer to make the kmer trainable.

    Returns
    -------
    np.array
        A length N array contains the boolean variable that mark the trainble kmers, N is the number of kmer.

    """
    trainable = np.ones(len(idx2kmer),dtype = bool)
    for i,kmer in enumerate(idx2kmer):
        for c in kmer:
            if c in exclude_bases:
                trainable[i] = False
                break
        for b in must_contain:
            if b not in kmer:
                trainable[i] = False
                break
    return trainable

def initialize_modified_kmers(idx2kmer:List, 
                              emission_means:np.array, 
                              shift:float = 0.0,
                              modified_bases:Dict = {'A':'M'})->np.array:
    """
    Generate the initialization of the modified kmers.

    Parameters
    ----------
    idx2kmer : List
        The index to kmer look-up list.
    emission_means : np.array
        The current mean of the kmers.
    shift : float, optional
        The prior shift for the modified kmer. The default is -0.3.
    modified_bases : Dict, optional
        The dictionary of alphabeta of all modified bases, default is {'M':'A'}.

    """
    modified_means = np.copy(emission_means)
    for i,kmer in enumerate(idx2kmer):
        if any([b in kmer for b in modified_bases.values()]):
            orig_kmer = kmer
            for key,val in modified_bases.items():
                orig_kmer = orig_kmer.replace(val,key)
            modified_means[i] = emission_means[idx2kmer.index(orig_kmer)]+shift
    return modified_means

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
                 "device":args.device,
                 "learning_rate":args.lr,
                 "weight_decay":args.weight_decay,
                 "momentum":args.momentum,
                 "exploration":args.exploration}
    train_config = TRAIN_CONFIG()
    MODIFIED_BASES = {"A":"M"}
    if args.trainable_bases is None:
        args.trainable_bases = "!"
    train_config.TRAIN["trainable_bases"] = args.trainable_bases
    ### Read dataset
    print("Load the dataset.")
    if 'base_prior' not in config.keys():
        config['base_prior'] = None
    if args.methylation_proportion is not None:
        config['base_prior'] = {"M":args.methylation_proportion,
                                "A":1-args.methylation_proportion}
        #TODO: this has to be changed latter to adapt to multi base-modifications.
    chunks = np.load(os.path.join(args.input,"chunks.npy"),mmap_mode = 'r')
    durations = np.load(os.path.join(args.input,"durations.npy"),mmap_mode = 'r')
    kmers = np.load(os.path.join(args.input,"kmers.npy"),mmap_mode = 'r')
    k2t = Kmer2Transition(config['alphabeta'],config['k'],config['chunk_len'],config['kmer2idx_dict'],config['idx2kmer'],base_alternation = MODIFIED_BASES, base_prior = config['base_prior'],kmer_replacement = args.kmer_replacement)
    dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
    loader = DataLoader(dataset,batch_size = args.batch_size, shuffle = True)
    loader = DeviceDataLoader(loader,device = args.device)
    
    ### Initialize the model
    print("Construct and initialize the model.")
    n_states = len(config['alphabeta'])**config['k']
    init_means = np.random.rand(n_states)-0.5
    init_covs = np.asarray([0.5]*n_states)
    N,L = chunks.shape
    norm = Normalizer(no_scale = True)
    duration_mask = np.repeat(np.arange(L)[None,:],N,axis = 0)<durations[:,None]
    if args.pretrain > 0:
        renorm_chunks = chunks
        for j in tqdm(np.arange(args.pretrain),desc = "Renorm the signal:"):
            rc_signal = np.asarray([[init_means[x].item() for x in kmers[i]] for i in tqdm(np.arange(chunks.shape[0]),desc = "Generate reconstruction signal.")])
            print("Renormalization.")
            renorm_chunks = norm(renorm_chunks,rc_signal, durations,kmers)
            print("Save the renormalized chunks.")
            np.save(os.path.join(args.input,"chunks_renorm_%d.npy"%(j)),renorm_chunks)
            chunks = renorm_chunks
    if not args.retrain:
        for i in tqdm(np.arange(n_states),desc = "Initialize parameter from the given chunks."):
            selected_sig = chunks[np.logical_and(duration_mask,kmers == i)]
            if len(selected_sig) > 0:
                init_means[i] = np.mean(selected_sig)
        np.save(os.path.join(args.model_folder,"initial_means0.npy"),init_means)
        print("Rescale the signal using the initialized parameters.")
    trainable = trainable_kmer(train_config.DATA['idx2kmer'],*train_config.TRAIN["trainable_bases"].split('!'))
    emission = GaussianEmissions(init_means[:,None], init_covs[:,None],trainable = trainable)
    hmm = RHMM(emission,normalize_transition=False,device = args.device,sparse_operation=True)
    
    ### Train
    trainer = RHMM_Trainer(loader,hmm,config = train_config)
    lr = args.lr
    epoches = args.epoches
    optim = train_config.TRAIN['optimizer'](hmm.parameters(),lr = lr)
    
    if args.retrain:
        print("Load pretrained model.")
        trainer.load(args.model_folder)
    if args.reinitialize:
        print("Initialize the modified kmers according to the current kmer model.")
        modified_means = initialize_modified_kmers(config['idx2kmer'],
                                                   trainer.hmm.emission.means.cpu().detach().numpy(),
                                                   modified_bases=MODIFIED_BASES)
        trainer.hmm.emission.reinitialize_means(torch.from_numpy(modified_means).to(args.device))
    print("Begin training the model.")
    trainer.train_EM(epoches,args.report,args.model_folder)
    # trainer.train(epoches,optim,args.report,args.model_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training RHMM model')
    parser.add_argument("-i","--input", type = str, required = True,
                        help = "Data folder contains the chunk, kmer sequence.")
    parser.add_argument('-o', '--model_folder', required = True,
                        help = "The folder to save folder at.")
    parser.add_argument("-b","--batch_size", type = int, default = 30,
                        help = "The batch size to train.")
    parser.add_argument("--lr",type = float, default = 1.0,
                        help = "The initial training learning rate.")
    parser.add_argument("--weight_decay",type = float, default = 0.0,
                        help = "The weight decay of the parameters.")
    parser.add_argument("--momentum",type = float, default = 0.9,
                        help = "The momentum value.")
    parser.add_argument("--exploration",type = float, default = 1.5,
                        help = "The factor of setting covariance to certainty, bigger number make the model explore more space of mean but may decrease the convergence speed.")
    parser.add_argument("--report",type = int, default = 1,
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
    parser.add_argument('--trainable_bases', type = str, default = None,
                        help="A magic string AB!CD or AB that gives the trainable bases and NOT trainalbe bases, for example MC means trains on kmer that must contains M and C, A!M means trains on kmer that must contains A but not contains M.")
    parser.add_argument('--methylation_proportion', type = float, default = None,
                        help="The expecting methylation proportion.")
    parser.add_argument('--pretrain', type = int, default = 0,
                        help="The rounds to renormalize signal and pretrain.")
    parser.add_argument('--initialize_modified_kmers', dest = 'reinitialize',
                        action = "store_true", help = "Reinitialize the modifed kmers.")
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.model_folder,exist_ok=True)
    train(args)
    