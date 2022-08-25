#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 07:09:45 2022

@author: heavens
"""
import os
import toml
import random
import itertools
from functools import partial
from tqdm import tqdm
import numpy as np


class LinkingGraph(object):
    def __init__(self,config):
        self.config = config
        self.idx2kmer = config['idx2kmer']
        self.kmer2idx = config['kmer2idx_dict']
        self.n_kmer = len(self.idx2kmer)
        self.nodes = {}
        self.invariant_kmers = []
        
    def build_invariant_kmers(self, variant_bases):
        for i in np.arange(self.n_kmer):
            if not any([x in self.idx2kmer[i] for x in variant_bases]):
                self.invariant_kmers.append(i)
    
    def add_reads(self,segments,decoded,min_seg_len = 50,transition = None):
        if len(self.invariant_kmers) == 0:
            raise ValueError("Call build_invariant_kmers first.")
        if transition is not None:
            decoded = np.asarray([[transition(x) for x in d] for d in decoded])
        mask = np.isin(decoded,self.invariant_kmers)
        breakings = np.where(mask)
        start,end,duration = self.boundering(breakings)
        for i,start_index in tqdm(enumerate(start[:,:-1].T),total = len(start)-1):
            s_i,s_j = start_index
            next_i = start[0][i+1]
            if next_i != s_i:
                continue
            if (start[1][i+1] - s_j) < min_seg_len:
                continue
            curr_kmer = decoded[s_i,s_j]
            curr_segment = segments[s_i,s_j:end[1][i+1]+1]
            curr_path = decoded[s_i,s_j:end[1][i+1]+1]
            if curr_kmer not in self.nodes.keys():
                node = KmerNode(curr_kmer)
                self.nodes[curr_kmer] = node
            self.nodes[curr_kmer].add_segment(curr_segment,curr_path)
    
    def boundering(self,breakings):
        x,y = breakings
        start_mask = np.logical_or((x[1:]-x[:-1]) != 0,(y[1:]-y[:-1]) != 1)
        start = np.asarray([np.insert(x[1:][start_mask],0,x[0]),np.insert(y[1:][start_mask],0,y[0])])
        end_mask = np.logical_or((x[:-1]-x[1:]) != 0,(y[:-1]-y[1:]) != -1)
        end = np.asarray([np.append(x[:-1][end_mask],x[-1]),np.append(y[:-1][end_mask],y[-1])])
        duration = [end[0] - start[0],end[1]-start[1]+1]
        assert np.all(duration[0] == 0)
        return start, end, duration[1]

    def sampling_once(self,max_len):
        segment = np.asarray([])
        path = np.asarray([],dtype = np.int16)
        kmer = random.choice(list(self.nodes.keys()))#random choose a kmer to start
        while len(segment)<max_len:
            node = self.nodes[kmer]
            curr_seg,curr_path,kmer = node.sampling()
            segment = np.concatenate((segment,curr_seg),axis = 0)
            path = np.concatenate((path,curr_path),axis = 0)
        return segment[:max_len],path[:max_len]
    
    def sampling(self,max_len,n_times):
        segments = []
        paths = []
        for i in tqdm(range(n_times),desc = "Sampling"):
            seg,path = self.sampling_once(max_len)
            segments.append(seg)
            paths.append(path)
        return np.stack(segments,axis = 0),np.stack(paths,axis = 0)
        
class KmerNode(object):
    def __init__(self,kmer):
        self.kmer = kmer
        self.segments = []
        self.head_stickers = []
        self.tail_stickers = []
        self.paths = []
        self.next_kmers = []
        self.next_nodes = []
        
    def add_segment(self,segment,path):
        """
        Add segement to the node, both the start and end kmer need to be
        included

        Parameters
        ----------
        segment : np.array
            The segment of the signal, the start and the end invariant kmer
            need to be included.
        path : np.array
            The corresponding kmer path.
        """
        assert len(segment) == len(path)
        self.segments.append(segment[path != path[-1]])
        self.head_stickers.append(segment[path == path[0]])
        self.tail_stickers.append(segment[path == path[-1]])
        self.paths.append(path[path != path[-1]])
        self.next_kmers.append(path[-1])
        
    def sampling(self):
        i = random.randint(0,len(self.segments)-1)
        return self.segments[i],self.paths[i],self.next_kmers[i]

def kmers2seq(kmers,idx2kmer):
    merged = [g for g,_ in itertools.groupby(kmers)]
    seqs = [idx2kmer[x][0] for x in merged]
    return ''.join(seqs) + idx2kmer[merged[-1]][1:]

def load_data(data_f,mmap_mode = "r"):
    chunks = np.load(os.path.join(data_f,"chunks.npy"),mmap_mode = mmap_mode)
    paths = np.load(os.path.join(data_f,"path.npy"),mmap_mode = mmap_mode)
    return chunks,paths

def max_loop(seq,max_loop_size = 10):
    repeat_size = []
    for loop_size in np.arange(2,max_loop_size):
        curr_size = []
        for indent in np.arange(loop_size):
            curr_size.append(np.max([len(list(g)) for k,g in itertools.groupby(seq.replace("M","A")[indent:][::loop_size])]))
        repeat_size.append(np.min(curr_size))
    return np.max(repeat_size)

def m2a(kmer_i,idx2kmer,kmer2idx):
    if 'M' not in idx2kmer[kmer_i]:
        return kmer_i
    else:
        return kmer2idx[idx2kmer[kmer_i].replace('M','A')]

def a2m(kmer_i,idx2kmer,kmer2idx):
    if 'A' not in idx2kmer[kmer_i]:
        return kmer_i
    else:
        return kmer2idx[idx2kmer[kmer_i].replace('A','M')]

if __name__ == "__main__":
    home_f = os.path.expanduser("~")
    control_folder = home_f + "/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_noise"
    m6a_folder = home_f + "/bridge_scratch/ime4_Yearst/IVT/m6A/rep2/kmers_guppy_4000_noise"
    sample_n = 100000
    mix_ratio = 1 #M/A ratio
    config = toml.load(os.path.join(control_folder,"config.toml"))
    chunks_control,path_control = load_data(control_folder)
    chunks_m6A,path_m6A = load_data(m6a_folder)
    if chunks_control.shape[0]*mix_ratio>chunks_m6A.shape[0]:
        shrink = int(chunks_m6A.shape[0]/mix_ratio)
        chunks_control = chunks_control[:shrink]
        path_control = path_control[:shrink]
    else:
        shrink = int(chunks_control.shape[0]*mix_ratio)
        chunks_m6A = chunks_m6A[:shrink]
        path_m6A = path_m6A[:shrink]
    m2a_p = partial(m2a,idx2kmer = config['idx2kmer'],kmer2idx = config['kmer2idx_dict'])
    a2m_p = partial(a2m,idx2kmer = config['idx2kmer'],kmer2idx = config['kmer2idx_dict'])
    lg = LinkingGraph(config)
    lg.build_invariant_kmers(['A','M'])
    print("Add control data (%d chunks) into the graph."%(chunks_control.shape[0]))
    lg.add_reads(chunks_control,path_control,transition = m2a_p)
    print("Add m6A data (%d chunks) into the graph.")
    lg.add_reads(chunks_m6A,path_m6A,transition = a2m_p)
    print("Sampling the data.")
    s,p = lg.sampling(chunks_control.shape[1],sample_n)
    seqs = np.asarray([kmers2seq(x,lg.idx2kmer) for x in tqdm(p,total = len(p),desc = "Transfer to sequence")])
    seqs_len = np.asarray([len(x) for x in seqs])
    loop_filter = np.asarray([max_loop(seq)<4 for seq in tqdm(seqs,total = len(seqs),desc = "apply filter")])
    s,p,seqs,seqs_len = s[loop_filter],p[loop_filter],seqs[loop_filter],seqs_len[loop_filter]
    print("%d samples left after filtering"%(len(s)))
    print("Saving the data")
    print("Maximum sequence length %d"%(np.max(seqs_len)))
    out_f = home_f + "/Training_Datasets/cross_link_%dpct/"%(int(mix_ratio/(1.+mix_ratio)))
    os.makedirs(out_f,exist_ok = True)
    np.save(os.path.join(out_f,"chunks.npy"),s)
    np.save(os.path.join(out_f,"path.npy"),p)
    np.save(os.path.join(out_f,"seqs"),seqs)
    np.save(os.path.join(out_f,"seq_lens.npy"),seqs_len)
    
