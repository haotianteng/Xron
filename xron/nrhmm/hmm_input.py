#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:21:00 2022

@author: haotian teng
"""
import os
import toml
import torch
import itertools
import torchvision
import numpy as np
import torch.utils.data as data
from xron.utils.seq_op import findall
from typing import List,Dict

### Test module ###
from time import time

class Kmer_Dataset(data.Dataset):
    def __init__(self,
                 chunks:np.array,
                 durations:np.array,
                 kmer_labels:np.array,
                 transform:torchvision.transforms.transforms.Compose=None):
        
        self.chunks = chunks
        self.duration = durations
        self.labels = kmer_labels
        self.transform = transform
                
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sample = {'signal': self.chunks[idx][:,None].astype(np.float32), 
                  'duration': self.duration[idx],
                  'labels': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
def flatten(nested_list:List[List])->List:
    """
    Flat a nested list

    Parameters
    ----------
    nested_list : List[List]
        A nested list.

    Returns
    -------
    List
        The flatten list.

    """
    return [x for sub_list in nested_list for x in sub_list]
            
class Kmer2Transition(object):
    """
    Convert string sequence to integer index array
    
    Parameters
    ----------
    alphabeta : str
        The base alphabeta combination.
    k: int
        The size of the knmer.
    T_max: int
        The maximum length of each sequence.
    kmer2idx: dict
        The dictionary link the kmer to the index.
    idx2kmer: List
        The list give the kmer.
    neighbour_kmer: int
        The number of neighbourhood kmer to be considered when building the 
        transition matrix, if 0 then will only consider the current kmer when
        building transition matrix.
    base_alternation: dict, optional
        The dictionary gives the methylation replacement of the base, e.g. 
        {'A':'MX'}, means base A can be two kinds of methylation bases M and X.
        Default is None, no replacement will be made.
    kmer_replacement: bool, optional
        If we consider kmer replacement during transferring to the transition
        matrix.
            
    """
    def __init__(self,alphabeta:str, \
                 k:int, 
                 T_max:int, 
                 kmer2idx:dict, 
                 idx2kmer:List, 
                 neighbour_kmer:int = 2, 
                 base_alternation:Dict[str, str] = None,
                 kmer_replacement = False):
        self.base_n = len(alphabeta)
        self.alphabeta = alphabeta
        self.k = k
        self.n_states = self.base_n**self.k
        self.T_max = T_max
        self.neighbour = neighbour_kmer
        self.kmer2idx = kmer2idx
        self.idx2kmer = idx2kmer
        self.base_alternation = base_alternation
        self._build_background_matrix()
        self._build_methylation_alternation()
        self.kmer_replacement = kmer_replacement
        
    def _build_background_matrix(self):
        """
        Build the background transition matrix, i.e AACAA -> ACAAA,ACAAC,ACAAG,
        ACAAT,...

        """
        bg_source, bg_target = [],[]
        for i in np.arange(self.n_states):
            kmer = self.idx2kmer[i]
            bg_source.append(i)
            bg_target.append(i)
            for base in self.alphabeta:
                new_kmer = kmer[1:]+base
                j = self.kmer2idx[new_kmer]
                bg_source.append(i)
                bg_target.append(j)
        self.bg = np.zeros((self.n_states,self.n_states))
        self.bg[bg_source,bg_target] = 1
        
    def _build_methylation_alternation(self):
        """
        Create the methylation kmer alternative dictionary, e.g. ACGMT:[ACGMT,
        ACGAT, MCGAT, MCGMT], all kmer is represented using its index.

        """
        self.kmer_alternation = {}
        for i in np.arange(self.n_states):
            kmer = self.idx2kmer[i]
            alternations = []
            for perm_kmer in self._alternative_kmer(kmer):
                alternations.append(self.kmer2idx[perm_kmer])
            self.kmer_alternation[i] = alternations
            
    def _alternative_kmer(self,kmer):
        """
        Return an iterator gives the alternative kmer of the current kmer, e.g.
        given the current kmer is 'AACCG', the alternative kmer list will be
        ['MMCCG','MACCG','AMCCG','AACCG'], all kmer will be transferred to its
        corresponding indexs.

        Parameters
        ----------
        kmer : str
            The current kmer.

        Yields
        ------
        str
            Iterate over all the alternative kmer (include the kmer itself).

        """
        for k,v in self.base_alternation.items():
            for x in v:
                kmer.replace(x,k)
            v += k
            locs = findall(kmer,k)
            for x in itertools.product(v,repeat = len(locs)):
                curr_kmer = [x for x in kmer]
                for i,l in enumerate(locs):
                    curr_kmer[l] = x[i]
                yield ''.join(curr_kmer)
    
    def legal_transition(self,kmer_list:List[int]) -> List[set]:
        """
        Return the legal transition term given a kmer list.

        Parameters
        ----------
        kmer_list : List[int]
            DESCRIPTION.

        Returns
        -------
        List[set]
            DESCRIPTION.

        """
    
    def kmer2transition(self,kmer_seq):
        transitions = []
        condensation = [(k,len(list(g))) for k,g in itertools.groupby(kmer_seq)]
        condensed_kmer_seqs = [x[0] for x in condensation]
        condensed_transitions = self._get_transition_pairs(condensed_kmer_seqs)
        for i,v in enumerate(condensation):
            kmer,period = v
            curr = flatten(condensed_transitions[max(0,i-self.neighbour):i+self.neighbour+1])
            source,target = list(zip(*curr))
            curr_transition = torch.sparse_coo_tensor([source,target], [1.0]*len(source),(self.n_states,self.n_states))
            transitions += [curr_transition]*period
        return transitions
    
    def _get_transition_pairs(self,condensed_kmer_seqs):
        if self.kmer_replacement:
            condensed_kmer_seqs = [self.kmer_alternation[x] for x in condensed_kmer_seqs]
        else:
            condensed_kmer_seqs = [[x] for x in condensed_kmer_seqs]
        transitions = [list(itertools.product(condensed_kmer_seqs[0],condensed_kmer_seqs[0]))]
        for i in np.arange(1,len(condensed_kmer_seqs)):
            curr_trans = list(itertools.product(condensed_kmer_seqs[i-1],condensed_kmer_seqs[i]))+ list(zip(condensed_kmer_seqs[i],condensed_kmer_seqs[i]))
            curr_trans = set([x for x in curr_trans if self.bg[x[0],x[1]]])
            transitions.append(curr_trans)
        return transitions
    
    def __call__(self,sample:Dict):
        kmer_seq = list(sample['labels'])
        return {key:(value if key != 'labels' else self.kmer2transition(kmer_seq)) for key, value in sample.items()}
        

if __name__ == "__main__":
    from torchvision import transforms
    # kmers_f = "/home/heavens/BRIDGE_SCRATCH/NA12878_RNA_IVT/xron_output/extracted_kmers"
    kmers_f = "/home/heavens/bridge_scratch/NA12878_RNA_IVT/xron_partial/extracted_kmers"
    chunks = np.load(os.path.join(kmers_f,"chunks.npy"), mmap_mode= 'r')
    durations = np.load(os.path.join(kmers_f,"durations.npy"), mmap_mode = 'r')
    kmers = np.load(os.path.join(kmers_f,"kmers.npy"),mmap_mode = 'r')
    config = toml.load(os.path.join(kmers_f,"config.toml"))
    
    ### Test Kmer2Transition class
    k2t = Kmer2Transition('ACGTM',5,4000,config['kmer2idx_dict'],config['idx2kmer'],base_alternation = {"A":"M"}, kmer_replacement = True)
    t = k2t.kmer2transition(kmers[0])
    
    ### Test data loader
    dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
    loader = data.DataLoader(dataset,batch_size = 50, shuffle = True)
    for i_batch,batch in enumerate(loader):
        print(batch['signal'].shape)
        print(batch['duration'].shape)
        print(batch['labels'])
        break
