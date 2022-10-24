#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:47:56 2022

@author: haotian
"""
import h5py
import os
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm


def seq2kmer(seq:str,k:int = 5):
    return [seq[x:x+k] for x,_ in enumerate(seq[:-k+1])]

prefix = "/home/heavens/bridge_scratch/NA12878_RNA_IVT/guppy_train/fast5s"
kmer_dict_p1 = defaultdict(list)
kmer_dict_m1 = defaultdict(list)
k = 5
for f in tqdm(os.listdir(prefix)):
    if f.endswith("fast5"):
        with h5py.File(os.path.join(prefix,f),'r') as root:
            for read in root:
                try:
                    corrected = root[read]['Analyses/Segmentation_000/Reference_corrected']
                    signal = np.asarray(root[read]['Raw/Signal'])[::-1]
                    seq =np.asarray(corrected['ref_seq']).item().decode()
                    kmers = seq2kmer(seq,k = k)
                    ref_sig_idx = np.asarray(corrected['ref_sig_idx'])
                    for i in np.arange(len(kmers)):
                        c,c_n = ref_sig_idx[i+k//2],ref_sig_idx[i+k//2+1]
                        kmer = kmers[i]
                        kmer_dict_p1[kmer]+= list(signal[c:c_n])
                    for i in np.arange(len(kmers)):
                        c,c_n = ref_sig_idx[i+k//2-1],ref_sig_idx[i+k//2]
                        kmer = kmers[i]
                        kmer_dict_m1[kmer]+= list(signal[c:c_n])
                        
                except KeyError:
                    pass
    

std_m1 = []
std_p1 = []
for key,val in kmer_dict_m1.items():
    std_m1.append(np.std(val))
    std_p1.append(np.std(kmer_dict_p1[key]))