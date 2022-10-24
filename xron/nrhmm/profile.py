#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 07:09:45 2022

@author: heavens
"""
import os
import csv
import toml
import itertools
from tqdm import tqdm
import numpy as np


class Profiler(object):
    def __init__(self,data_f, config):
        self.data_f = data_f
        print("Loading chunk data")
        chunks_f = os.path.join(data_f,"chunks.npy")
        self.chunks = np.load(chunks_f)
        self.chunk_flatten = self.chunks.flatten()
        print("Loading decoded path data")
        decoded_f = os.path.join(data_f,"path.npy")
        self.decoded = np.load(decoded_f)
        self.decoded_flatten = self.decoded.flatten()
        self.config = config
        self.idx2kmer = config['idx2kmer']
        self.kmer2idx = config['kmer2idx_dict']
        self.n_kmer = len(self.idx2kmer)
        self.alphabeta = config['alphabeta']
        self.n_base = len(self.alphabeta)
        self.transitions_count = [[[]for _ in np.arange(self.n_base)] for _ in np.arange(self.n_kmer)]
        self.transition_weight = np.zeros((self.n_kmer,self.n_base))
        self.epsilon = 1e-6
        assert np.equal(self.chunks.shape,self.decoded.shape).all()
        group_f = os.path.join(self.data_f,"grouped.npy")
        if os.path.isfile(group_f):
            print("Loading grouped array.")
            self.kmer_groups = np.load(group_f,allow_pickle = True)
        else:
            self.kmer_groups = None

    def build_invariant_kmers(self, variant_bases):
        self.invariant_kmers = []
        for i in np.arange(self.n_kmer):
            if not any([x in self.idx2kmer[i] for x in variant_bases]):
                self.invariant_kmers.append(i)
            
    def grouping(self):
        print("Sorting decoded path")
        argsort = np.argsort(self.decoded_flatten)
        sorted_chunks = self.chunk_flatten[argsort]
        print("Grouping signal")
        kmers,idxs = np.unique(self.decoded_flatten[argsort],return_index = True)
        grouped = np.split(sorted_chunks,idxs[1:])
        self.kmer_groups = [[] for _ in np.arange(self.n_kmer)]     
        for i,kmer in enumerate(kmers):
            self.kmer_groups[kmer] = grouped[i]
        print("Writing summary")
        np.save(os.path.join(self.data_f,"grouped.npy"),self.kmer_groups)

    def summarize(self):
        if self.kmer_groups is None:
            raise ValueError("Grouping file has not been found, please run grouping first.")
        self.means = np.asarray([np.mean(self.kmer_groups[i]) for i in tqdm(np.arange(self.n_kmer))])
        self.stds = np.asarray([np.std(self.kmer_groups[i]) for i in tqdm(np.arange(self.n_kmer))])
        
    def summarize_length(self):
        flatten_path = self.decoded.flatten()
        self.kmer_length = {}
        for i,kmer in tqdm(enumerate(self.idx2kmer)):
            condition = flatten_path == i
            self.kmer_length[kmer] = np.diff(np.where(np.concatenate(([condition[0]],
                                                                      condition[:-1] != condition[1:],
                                                                      [True])))[0])[::2]
    
    def summarize_transition(self):
        for path in self.decoded:
            keys,counts = [],[]
            for key,group in itertools.groupby(path):
                keys.append(key)
                counts.append(len(list(group)))
            for i,k in enumerate(keys):
                if i == len(keys) - 1:
                    break
                next_base = self.idx2kmer[keys[i+1]][-1]
                self.transitions_count[k][self.alphabeta.index(next_base)].append(counts[i])
        for i in np.arange(self.n_kmer):
            for j in np.arange(self.n_base):
                c = self.transitions_count[i][j]
                self.transition_weight[i][j] = len(c)/(sum(c)+self.epsilon) #ML estimation of transition probability is N/sum_i k_i - N respect that the stay probability is 1, see the notebook for more details.
    
if __name__ == "__main__":
    home_f = os.path.expanduser("~")
    control_folder = home_f + "/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_noise"
    m6a_folder = home_f + "/bridge_scratch/ime4_Yearst/IVT/m6A/rep2/kmers_guppy_4000_noise"
    config = toml.load(os.path.join(control_folder,"config.toml"))
    p_control= Profiler(control_folder,config)
    # p_control.summarize()
    # p_control.summarize_length()
    p_control.summarize_transition()
    p_m6A= Profiler(m6a_folder,config)
    # p_m6A.summarize()
    # p_m6A.summarize_length()
    p_m6A.summarize_transition()
    
        