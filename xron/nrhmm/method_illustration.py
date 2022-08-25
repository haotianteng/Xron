"""
Created on Wed May 11 02:05:35 2022

@author: Haotian Teng
"""
import os
import time
import toml
import torch
import itertools
import numpy as np
from matplotlib import pyplot as plt
from xron.nrhmm.hmm import GaussianEmissions, RHMM
from xron.nrhmm.hmm_input import Kmer2Transition, Kmer_Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from xron.xron_train_base import DeviceDataLoader

def kmers2seq(kmers,idx2kmer):
    merged = [g for g,_ in itertools.groupby(kmers)]
    seqs = [idx2kmer[x][2] for x in merged]
    return ''.join(seqs)

torch.manual_seed(1992)
model = torch.load("/home/heavens/bridge_scratch/xron_rhmm_models_new/ckpt-12855")
emission = GaussianEmissions(model['hmm']['emission.means'].cpu().numpy(), 1*np.ones(3125)[:,None])
class TestArguments:
    input = "/home/heavens/bridge_scratch/NA12878_RNA_IVT/xron_partial/extracted_kmers/"
    # input = "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_90_pct/20210430_1745_X2_FAQ15454_23428362/kmers_guppy_4000/"
    batch_size = 10
    device = "cuda"
args = TestArguments
hmm = RHMM(emission,normalize_transition=False,device = args.device)
config = toml.load(os.path.join(args.input,"config.toml"))
print("Readout the pore model.")
chunks = np.load(os.path.join(args.input,"chunks.npy"))
n_samples, sig_len = chunks.shape
durations = np.load(os.path.join(args.input,"durations.npy"))
idx2kmer = config['idx2kmer']
kmers = np.load(os.path.join(args.input,"kmers.npy"))
k2t = Kmer2Transition(alphabeta = config['alphabeta'],
                      k = config['k'],
                      T_max = config['chunk_len'],
                      kmer2idx = config['kmer2idx_dict'],
                      idx2kmer = config['idx2kmer'],
                      neighbour_kmer = 2,
                      base_alternation = {"A":"M"}, 
                      kmer_replacement = True)
dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
loader = DataLoader(dataset,batch_size = args.batch_size, shuffle = True)
loader = DeviceDataLoader(loader,device = args.device)
for i_batch, batch in enumerate(loader):
    signal_batch = batch['signal']
    duration_batch = batch['duration']
    transition_batch = batch['labels']
    kmers_batch = batch['kmers']
    Ls = duration_batch.cpu().numpy()
    break
idx = 6
signal = signal_batch[idx].detach().cpu().numpy()[:400]
kmers = kmers_batch[idx].detach().cpu().numpy()[:400]
seq = kmers2seq(kmers,idx2kmer)

## Print the alignment
fig,axs = plt.subplots(nrows = 2, 
                       ncols = 2, 
                       gridspec_kw={'width_ratios': [1, 10],
                                    "height_ratios":[1,5],
                                    'wspace':0, 
                                    'hspace':0},
                       figsize = (6,6))
for i, ax in enumerate(fig.axes):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
axs[0][0].set_axis_off()
axs[0][1].plot(signal)
axs[0][1].xaxis.set_visible(False)
for i,c in enumerate(seq):
    axs[1][0].text(x = 0.4,y = i/len(seq)+0.01,s = c, fontsize = 10)
alignment = []
i=0
for x,g in itertools.groupby(kmers):
    alignment += [i]*len(list(g))
    i+=1
alignment = np.asarray(alignment)
axs[1][1].plot(alignment[-1] - alignment,color = "black")
axs[1][1].set_ylim(ymin = -0.5,ymax = alignment[-1]+0.5)

## Print the Marcus suggestion
fig,axs = plt.subplots(nrows = 2, 
                       ncols = 2, 
                       gridspec_kw={'width_ratios': [1, 10],
                                    "height_ratios":[1,5],
                                    'wspace':0, 
                                    'hspace':0},
                       figsize = (6,6))
for i, ax in enumerate(fig.axes):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
axs[0][0].set_axis_off()
axs[0][1].plot(signal)
axs[0][1].xaxis.set_visible(False)
for i,c in enumerate(seq):
    axs[1][0].text(x = 0.4,y = i/len(seq)+0.01,s = c, fontsize = 10)
alignment = []
i=0
for x,g in itertools.groupby(kmers):
    alignment += [i]*len(list(g))
    i+=1
alignment = np.asarray(alignment)
axs[1][1].plot(alignment[-1] - alignment,color = "black")
axs[1][1].set_ylim(ymin = -0.5,ymax = alignment[-1]+0.5)
axs[1][1].fill_between(x = np.arange(len(signal)),
                       y1 = -0.5,
                       y2 = alignment[-1]+0.5,
                       color = "grey")

## Print the Marcus suggestion
fig,axs = plt.subplots(nrows = 2, 
                       ncols = 2, 
                       gridspec_kw={'width_ratios': [1, 10],
                                    "height_ratios":[1,5],
                                    'wspace':0, 
                                    'hspace':0},
                       figsize = (6,6))
for i, ax in enumerate(fig.axes):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
axs[0][0].set_axis_off()
axs[0][1].plot(signal)
axs[0][1].xaxis.set_visible(False)
for i,c in enumerate(seq):
    axs[1][0].text(x = 0.4,y = i/len(seq)+0.01,s = c, fontsize = 10)
alignment = []
i=0
for x,g in itertools.groupby(kmers):
    alignment += [i]*len(list(g))
    i+=1
alignment = np.asarray(alignment)
alignment = alignment[-1] - alignment
axs[1][1].plot(alignment,color = "black")
axs[1][1].fill_between(x = np.arange(len(signal)),
                       y1 = alignment - 3,
                       y2 = alignment + 3,
                       color = "grey")
axs[1][1].set_ylim(ymin = -0.5,ymax = alignment[0]+0.5)