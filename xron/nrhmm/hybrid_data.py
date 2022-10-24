#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 07:09:45 2022

@author: heavens
"""
import os
import sys
import toml
import random
import argparse
import itertools
from functools import partial
from tqdm import tqdm
import numpy as np


class LinkingGraph(object):
    def __init__(self,config):
        self.config = config
        self.idx2kmer = config['idx2kmer']
        self.kmer2idx = config['kmer2idx_dict']
        self.exploration = config['exploration']
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

    def sampling_once(self,max_len,add_noise = 0):
        segment = np.asarray([])
        path = np.asarray([],dtype = np.int16)
        p = np.asarray([x.weight for x in self.nodes.values()])
        if np.random.random() < exploration:
            kmer = np.random.choice(list(self.nodes.keys()),p = p/p.sum())#random choose a kmer to start
        else:
            kmer = list(self.nodes.keys())[np.argmax(p)]
        while len(segment)<max_len:
            node = self.nodes[kmer]
            curr_seg,curr_path,kmer = node.sampling(exploration = self.exploration)
            segment = np.concatenate((segment,curr_seg),axis = 0)
            path = np.concatenate((path,curr_path),axis = 0)
        if add_noise > 0:
            segment += np.random.normal(loc = 0.0,scale = add_noise,size = len(segment))
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
        self.visits = [] #Sampling times
        self.weight = 0.
        self.epsilon = 1e-6
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
        self.visits.append(self.epsilon)
        self.weight += 1
        
    def sampling(self,exploration = 0.1):
        p = np.sqrt(1/np.asarray(self.visits)) #UCB score.
        p = p/sum(p)
        if np.random.random() < exploration:
            i = np.random.choice(len(self.segments), p=p)
        else:
            i = np.argmax(p)
        self.visits[i] += 1
        self.weight = len(self.visits)/np.sqrt(sum(self.visits))
        return self.segments[i],self.paths[i],self.next_kmers[i]

def kmers2seq(kmers,idx2kmer):
    merged = [g for g,_ in itertools.groupby(kmers)]
    seqs = [idx2kmer[x][0] for x in merged]
    return ''.join(seqs) + idx2kmer[merged[-1]][1:]

def load_data(data_f,mmap_mode = "r",max_n = None,shuffle = False):
    chunks = np.load(os.path.join(data_f,"chunks.npy"),mmap_mode = mmap_mode)
    paths = np.load(os.path.join(data_f,"path.npy"),mmap_mode = mmap_mode)
    if max_n is not None:
        chunks = chunks[:max_n]
        paths = paths[:max_n]
    if shuffle:
        perm = np.arange(len(chunks))
        perm = np.random.permutation(perm)
        chunks = chunks[perm]
        paths = paths[perm]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--control","-c",required = True,type = str,help = "The control foldaer.")
    parser.add_argument("--modified","-m",required = True,type = str,help = "The modification folder.")
    parser.add_argument("--output","-o",required = True,type = str,help = "The output folder.")
    parser.add_argument("--sample_n","-n",type = int,default = 100000,help = "The number of sampling.")
    parser.add_argument("--mixture","-mix",type = float,default = 0.5,help = "The mixture ratio of the control and the modified.")
    parser.add_argument("--exploration","-e",type = float,default = 0.1,help = "The exploration ratio.")
    parser.add_argument("--max",type = int,default = None,help = "The maximum number of segments to load.")
    args = parser.parse_args(sys.argv[1:])

    ##Testing code
    # home_f = os.path.expanduser("~")
    # scratch_f = home_f + "/bridge_scratch"
    # control_folder = scratch_f + "/Training_Datasets/EVA+NA12878IVT+ELIGOS+ONTBACTERIAMIX/control/"
    # modified_folder = scratch_f + "/Training_Datasets/EVA+NA12878IVT+ELIGOS+ONTBACTERIAMIX/modified/"
    # sample_n = 500000
    # max_n = None #For testing.
    # mix_ratio = 0.333 #M/A ratio
    # exploration = 0.2 #The decay of weight of edge after being samplinged.
    # out_f = home_f + "/Training_Datasets/cross_link_%dpct/"%(int(100*mix_ratio/(1.+mix_ratio)))

    ##Running code
    control_folder = args.control
    modified_folder = args.modified
    sample_n = args.sample_n
    mix_ratio = args.mixture
    exploration = args.exploration
    out_f = args.output
    max_n = args.max
    
    config = toml.load(os.path.join(control_folder,"config.toml"))
    config['exploration'] = exploration
    m2a_p = partial(m2a,idx2kmer = config['idx2kmer'],kmer2idx = config['kmer2idx_dict'])
    a2m_p = partial(a2m,idx2kmer = config['idx2kmer'],kmer2idx = config['kmer2idx_dict'])
    lg = LinkingGraph(config)
    lg.build_invariant_kmers(['A','M'])
    
    chunks_control,path_control = load_data(control_folder,max_n = max_n,shuffle = True)
    chunks_modified,path_modified = load_data(modified_folder,max_n = max_n,shuffle = True)
    if chunks_control.shape[0]*mix_ratio>chunks_modified.shape[0]:
        shrink = int(chunks_modified.shape[0]/mix_ratio)
        chunks_control = chunks_control[:shrink]
        path_control = path_control[:shrink]
    else:
        shrink = int(chunks_control.shape[0]*mix_ratio)
        chunks_modified = chunks_modified[:shrink]
        path_modified = path_modified[:shrink]
    print("Add control data (%d chunks) into the graph."%(chunks_control.shape[0]))
    lg.add_reads(chunks_control,path_control,transition = m2a_p)
    print("Add modified data (%d chunks) into the graph.")
    lg.add_reads(chunks_modified,path_modified,transition = a2m_p)
    print("Sampling the data.")
    s,p = lg.sampling(chunks_control.shape[1],sample_n)
    seqs = np.asarray([kmers2seq(x,lg.idx2kmer) for x in tqdm(p,total = len(p),desc = "Transfer to sequence")])
    seqs_len = np.asarray([len(x) for x in seqs])
    loop_filter = np.asarray([max_loop(seq)<4 for seq in tqdm(seqs,total = len(seqs),desc = "apply filter")])
    s,p,seqs,seqs_len = s[loop_filter],p[loop_filter],seqs[loop_filter],seqs_len[loop_filter]
    print("%d samples left after filtering"%(len(s)))
    print("Saving the data")
    print("Maximum sequence length %d"%(np.max(seqs_len)))
    os.makedirs(out_f,exist_ok = True)
    np.save(os.path.join(out_f,"chunks.npy"),s)
    np.save(os.path.join(out_f,"path.npy"),p)
    np.save(os.path.join(out_f,"seqs"),seqs)
    np.save(os.path.join(out_f,"seq_lens.npy"),seqs_len)
    
