"""
Created on Sat Mar 26 14:24:27 2022

@author: Haotian Teng
"""

import h5py
from xron.utils import seq_op
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from typing import List
        
fast5_f = "/home/heavens/Documents/FAQ15457_pass_6cc85380_1.fast5"
root = h5py.File(fast5_f,mode = 'r')
k = 5
strides = 11
for read_id in root:
    read_h = root[read_id]
    basecalled = read_h['Analyses/Basecall_1D_001/BaseCalled_template']
    hidden = np.asarray(basecalled['Hidden'])
    signal = np.asarray(read_h['Raw/Signal'])
    move = basecalled['Move']
    position_sig = np.repeat(np.cumsum(move)-1,repeats = strides).astype(np.int32)[:len(signal)]
    segmentation = np.where(np.diff(position_sig))[0]
    position = np.cumsum(move)
    fastq = np.asarray(basecalled['Fastq']).item().decode().split()[1]
    kmers_list = seq_op.kmers2array(seq_op.seq2kmers(fastq,k = k))
    # kmers_list = [0]*((k-1)//2) + kmers_list + [0]*((k-1)//2)
    kmers_list = [0]*(k-1) + kmers_list
    logit = 1/(1+1/np.exp(hidden))
    
    ## Mask to display partail
    max_pos = 10
    hidden = hidden[position<max_pos,:]
    logit = logit[position<max_pos,:]
    position = position[position<max_pos]
    
    size = 4
    umap = UMAP(n_components=2)
    tsne = TSNE(n_components=2, init = 'random')
    hidden_umap = umap.fit_transform(hidden)
    hidden_tsne = tsne.fit_transform(hidden)
    logit_umap = umap.fit_transform(logit)
    logit_tsne = tsne.fit_transform(logit)
    # cla = [kmers_list[x-1] for x in position]
    cla = position
    
    plt.figure(figsize = (80,40))
    max_T = 2000
    plt.plot(signal[::-1][:max_T],color = "red")
    plt.vlines(x = segmentation[segmentation<max_T], ymin = min(signal), ymax = max(signal), color = "green", linestyles = "dotted", linewidth = 0.01, antialiased=False)
    plt.figure()
    scatter = plt.scatter(hidden_umap[:,0],hidden_umap[:,1],s = size,c = cla, cmap='viridis')
    plt.title("hidden umap")
    plt.legend(handles=scatter.legend_elements()[0], labels=list(np.arange(max(cla+1))))
    
    plt.figure()
    scatter = plt.scatter(hidden_tsne[:,0],hidden_tsne[:,1],s = size,c = cla, cmap='viridis')
    plt.title("hidden TSNE")
    plt.legend(handles=scatter.legend_elements()[0], labels=list(np.arange(max(cla+1))))
    
    plt.figure()
    scatter = plt.scatter(logit_umap[:,0],logit_umap[:,1],s = size,c = cla, cmap='viridis')
    plt.title("logit umap")
    plt.legend(handles=scatter.legend_elements()[0], labels=list(np.arange(max(cla+1))))
    
    plt.figure()
    scatter = plt.scatter(logit_tsne[:,0],logit_tsne[:,1],s = size,c = cla, cmap='viridis')
    plt.title("logit TSNE")
    plt.legend(handles=scatter.legend_elements()[0], labels=list(np.arange(max(cla+1))))
    break