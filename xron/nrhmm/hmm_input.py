#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:21:00 2022

@author: haotian teng
"""
import os
import copy
import toml
import torch
import itertools
import torchvision
import numpy as np
import torch.utils.data as data
from xron.utils.seq_op import findall
from typing import List,Dict,Callable

### Test module ###
from time import time

class Kmer_Dataset(data.Dataset):
    def __init__(self,
                 chunks:np.array,
                 durations:np.array,
                 kmer_labels:np.array,
                 labels:np.array = None,
                 transform:torchvision.transforms.transforms.Compose=None):
        
        self.chunks = chunks
        self.duration = durations.astype(np.int64)
        self.labels = kmer_labels
        self.transform = transform
        self.y = labels
                
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sample = {'signal': self.chunks[idx][:,None].astype(np.float32), 
                  'duration': self.duration[idx],
                  'labels': self.labels[idx]}
        if self.y is not None:
            sample['y'] = self.y[idx]
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

class Normalizer(object):
    def __init__(self,
                 use_dwell:bool = True,
                 statistics:Callable = None,
                 no_scale:bool = False,
                 min_dwell:int = 5,
                 effective_kmers:List[int] = None):
        self.use_dwell = use_dwell
        self.statistics = np.median if statistics is None else statistics
        self.no_scale = no_scale
        self.min_dwell = min_dwell
        self.effective_kmers = effective_kmers
    def __call__(self,
                 *args,
                 **kwargs):
        if self.use_dwell:
            return self.rescale_dwell(*args, **kwargs)
        else:
            return self.rescale(*args, **kwargs)
        
    
    def dwell(self,
              sig:np.array,
              path:np.array,
              duration:int):
        """
        Return the dwell property of the signal
    
        Parameters
        ----------
        sig : np.array, shape [N].
            The original signal.
        path : np.array, shape [N]
            The decoded path of the current signal.
        duration : int
            The duration of the signal. The default is None.
        statistics: func, optional, default is median.
            The statistics of the dwell to return.
        min_dwell: int, optional, default is 3.
            The minimum length of the dwell to be considered.
        Returns
        -------
        TYPE
            Return the chosen statistics of the dwells.
    
        """
        gp = [list(y) for x,y  in itertools.groupby(zip(sig[:duration],path[:duration]),key = lambda x: x[1])]
        if self.effective_kmers is None:
            dwells = [[y[0] for y in x] for x in gp if len(x)>=self.min_dwell]
        else:
            dwells = [[y[0] for y in x] for x in gp if len(x)>=self.min_dwell and x[0][1] in self.effective_kmers]
            if len(dwells) < 5:
                # print("Warning, there are very few dwells obtained from effective kmers, whole kmers list will be used instead.")
                # dwells = [[y[0] for y in x] for x in gp if len(x)>=self.min_dwell]
                dwells = [[0.]]
        return [self.statistics(x) for x in dwells]
        
    def rescale(self,
                sig:np.array,
                rc_sig:np.array,
                duration:np.array,
                return_sig:bool = True):
        """
        Rescale the input signal using the reconstructed signal.
    
        Parameters
        ----------
        sig : np.array, shape [B,N].
            The original signal.
        rc_sig : np.array, shape [B,N]
            The reconstructed signal.
        duration : np.array, optional, shape [B]
            The duration of the signal batch. The default is None.
        return_sig: bool, optional, default is True.
            If return the normalized signal or scale.
        Returns
        -------
        TYPE
            Return the rescaled signal.
    
        """
        B,L = sig.shape
        if sig.ndim == 1:
            sig = sig[None,:]
        if rc_sig.ndim == 1:
            rc_sig = rc_sig[None,:]
        if duration is None:
            N = np.asarray([len(sig)]*sig.shape[0])
        else:
            if type(duration) == type(1):
                duration = np.asarray([duration])
            N = duration
        if self.no_scale:
            scale = np.asarray([1])
        else:
            scale = (N*np.sum(sig*rc_sig,axis = 1) - np.sum(sig,axis = 1)*np.sum(rc_sig,axis = 1))/(N*np.sum(sig**2,axis = 1)-np.sum(sig,axis = 1)**2)
        for i,s in enumerate(rc_sig):
            s[duration[i]:] = 0
        shift = np.mean(rc_sig,axis = 1)*L/duration-scale*np.mean(sig,axis = 1)*L/duration
        if return_sig:
            return scale[:,None]*sig + shift[:,None]
        else:
            return scale, shift
    
    def rescale_dwell(self,
                      sig:np.array,
                      rc_sig:np.array,
                      duration:np.array,
                      path:np.array):
        """https://researchcomputing.princeton.edu/support/knowledge-base/singularity
        Rescale the input signal using the reconstructed signal.
    
        Parameters
        ----------
        sig : np.array, shape [B,N].
            The original signal.
        rc_sig : np.array, shape [B,N]
            The reconstructed signal.
        duration : np.array, shape [B]
            The duration of the signal batch. The default is None.
        path : np.array, shape [B,N]
            The decoded path of the signal.
        Returns
        -------
        TYPE
            Return the rescaled signal.
    
        """
        dwell_sig = [self.dwell(s,p,d) for s,p,d in zip(sig,path,duration)]
        dwell_rc = [self.dwell(s,p,d) for s,p,d in zip(rc_sig,path,duration)]
        ls = [len(x) for x in dwell_sig]
        max_len = max(ls)
        dwell_sig_pad = [x + [0]*(max_len - len(x)) for x in dwell_sig]
        dwell_rc_pad = [x + [0]*(max_len - len(x)) for x in dwell_rc]
        scale,shift = self.rescale(np.array(dwell_sig_pad),
                                   np.array(dwell_rc_pad),
                                   np.array(ls),
                                   return_sig = False)
        return scale[:,None]*sig + shift[:,None]
    

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
    base_prior: Dict, optional
        The prior probability of each base, this is useful when fitting in-vitro
        transcription dataset, default is None, that every base has equal prior
        (1).
    kmer_replacement: bool, optional
        If we consider kmer replacement during transferring to the transition
        matrix.
            
    """
    def __init__(self,alphabeta:str, 
                 k:int, 
                 T_max:int, 
                 kmer2idx:dict, 
                 idx2kmer:List, 
                 neighbour_kmer:int = 2, 
                 base_alternation:Dict[str, str] = None,
                 base_prior = None,
                 kmer_replacement = False,
                 out_format = "sparse"):
        self.n_base = len(alphabeta)
        self.alphabeta = alphabeta
        self.k = k
        self.n_states = self.n_base**self.k
        self.T_max = T_max
        self.neighbour = neighbour_kmer
        self.kmer2idx = kmer2idx
        self.idx2kmer = idx2kmer
        self.base_alternation = base_alternation
        if base_prior is None:
            self.base_prior = {x:1 for x in self.alphabeta}
        else:
            self.base_prior = base_prior
            for b in self.alphabeta:
                if b not in self.base_prior.keys():
                    self.base_prior[b] = 1
        self._build_background_matrix()
        self._build_methylation_alternation()
        self.kmer_replacement = kmer_replacement
        self.out_format = out_format
        
    def _build_background_matrix(self):
        """
        Build the background transition matrix, i.e AACAA -> ACAAA,ACAAC,ACAAG,
        ACAAT,...

        """
        bg_source, bg_target = [],[]
        bg_idx = []
        self.from_idx = np.zeros((self.n_states,self.n_base+1))
        bg_vals = []
        for i in np.arange(self.n_states):
            kmer = self.idx2kmer[i]
            bg_source.append(i)
            bg_target.append(i)
            bg_vals.append(1)
            curr_tgt = [i]
            for base in self.alphabeta:
                new_kmer = kmer[1:]+base
                j = self.kmer2idx[new_kmer]
                if i==j:
                    curr_tgt.append(j)
                    continue
                bg_source.append(i)
                bg_target.append(j)
                bg_vals.append(self.base_prior[base])
                curr_tgt.append(j)
            bg_idx.append(curr_tgt)
        self.bg = np.zeros((self.n_states,self.n_states),dtype = np.float32)
        self.bg[bg_source,bg_target] = np.array(bg_vals)
        self.to_idx = np.array(bg_idx)
        for i in np.arange(self.n_states):
            kmer = self.idx2kmer[i]
            self.from_idx[i][0] = i
            for j,base in enumerate(self.alphabeta):
                new_kmer = base + kmer[:-1]
                new_idx = self.kmer2idx[new_kmer]
                self.from_idx[i][j+1] = new_idx
        self.idx_map = [self.from_idx,self.to_idx]
        
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
                kmer = kmer.replace(x,k)
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
        # kmer_seq = copy.deepcopy(kmer_seq) #To avoid torch.dataLoader blowing up shared memory according to this issue:https://github.com/pytorch/pytorch/issues/11201
        condensation = [(k,len(list(g))) if k>=0 else (0,len(list(g))) for k,g in itertools.groupby(kmer_seq)]
        condensed_kmer_seqs = [x[0] for x in condensation]
        condensed_transitions = self._get_transition_pairs(condensed_kmer_seqs)
        for i,(kmer,period) in enumerate(condensation):
            curr = flatten(condensed_transitions[max(0,i-self.neighbour):i+self.neighbour+1])
            source,target = list(zip(*curr))
            curr_transition = torch.sparse_coo_tensor([source,target], self.bg[source,target],(self.n_states,self.n_states))
            transitions += [curr_transition]*period
        return transitions
    
    def kmer2compact_transition(self,kmer_seq):
        transitions = []
        condensation = [(k,len(list(g))) if k>=0 else (0,len(list(g))) for k,g in itertools.groupby(kmer_seq)]
        condensed_kmer_seqs = [x[0] for x in condensation]
        condensed_transitions = self._get_transition_pairs(condensed_kmer_seqs)
        for i,(kmer,period) in enumerate(condensation):
            curr = flatten(condensed_transitions[max(0,i-self.neighbour):i+self.neighbour+1])
            transition = self.compact_transition_tensor(curr, [self.bg[s,t] for s,t in curr])
            transitions += [transition]*period
        return transitions
    
    def compact_transition_tensor(self,src_tgt:List,value:List) -> torch.tensor:
        """
        Generate a compact transition tensor given the soruce-target paris and
        the values.

        Parameters
        ----------
        src_tgt : List with shape [L]
            A List contains the source and target tuple.
        value : List with shape [L]
            List contains the values, this list should have the same length as
            the source and target.

        Returns
        -------
        A compact torch tensor with shape [N,B+1], where N is the number of kmers
        and B is the number of bases, and 0 index always stand for kmers stay.

        """
        trs = torch.zeros((self.n_states,self.n_base + 1),dtype = torch.float)
        trt = torch.zeros((self.n_base + 1,self.n_states),dtype = torch.float)
        for (src,tgt),val in zip(src_tgt,value):
            trs[src][np.where(self.to_idx[src] == tgt)[0][0]] += val
        for (src,tgt),val in zip(src_tgt,value):
            trt[np.where(self.from_idx[tgt]==src)[0][0]][tgt] += val
        return torch.cat((trs,trt.T),dim = 1)
        
    
    def _get_transition_pairs(self,condensed_kmer_seqs):
        if self.kmer_replacement:
            condensed_kmer_seqs = [self.kmer_alternation[x] for x in condensed_kmer_seqs]
        else:
            condensed_kmer_seqs = [[x] for x in condensed_kmer_seqs]
        transitions = []
        for i in np.arange(0,len(condensed_kmer_seqs)):
            if i == 0:
                curr_trans = list(zip(condensed_kmer_seqs[i],condensed_kmer_seqs[i]))
            else:
                curr_trans = list(itertools.product(condensed_kmer_seqs[i-1],condensed_kmer_seqs[i]))+ list(zip(condensed_kmer_seqs[i],condensed_kmer_seqs[i]))
            curr_trans = set([x for x in curr_trans if self.bg[x[0],x[1]]])
            transitions.append(curr_trans)
        return transitions
    
    def __call__(self,sample:Dict):
        func = self.kmer2transition if self.out_format == "sparse" else self.kmer2compact_transition
        transformed = {key:(value if key != 'labels' else func(value)) for key, value in sample.items()}
        transformed['kmers'] = sample['labels']
        return transformed
        

if __name__ == "__main__":
    from torchvision import transforms
    torch.multiprocessing.set_sharing_strategy('file_system')
    import time
    # kmers_f = "/home/heavens/BRIDGE_SCRATCH/NA12878_RNA_IVT/xron_output/extracted_kmers"
    kmers_f = "/home/haotian/bridge_scratch/NA12878_RNA_IVT/xron_partial/extracted_kmers"
    chunks = np.load(os.path.join(kmers_f,"chunks.npy"))
    durations = np.load(os.path.join(kmers_f,"durations.npy"))
    kmers = np.load(os.path.join(kmers_f,"kmers.npy"))
    config = toml.load(os.path.join(kmers_f,"config.toml"))
    
    ### Test Kmer2Transition class
    k2t = Kmer2Transition('ACGTM',5,4000,config['kmer2idx_dict'],config['idx2kmer'],base_alternation = {"A":"M"}, kmer_replacement = True, out_format = "compact")
    t = k2t.kmer2transition(kmers[0])
    
    ### Test data loader
    dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
    loader = data.DataLoader(dataset,batch_size = 2, shuffle = True,num_workers=2)
    start = time.time()
    for i_batch,batch in enumerate(loader):
        print("Batch loading time %.2f"%(time.time() - start))
        time.sleep(3)
        start = time.time()
        # print(batch['signal'].shape)
        # print(batch['duration'].shape)
        # print(batch['labels'])
