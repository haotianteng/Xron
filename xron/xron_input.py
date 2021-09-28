"""
Created on Thu Mar  4 19:33:04 2021

@author: Haotian Teng
"""
import numpy as np
import torchvision
import torch
from torch import nn
import torch.utils.data as data
from torchvision import transforms
from matplotlib import pyplot as plt
from typing import Dict,Callable
from functools import partial

RNA_FILTER_CONFIG = {"min_rate":10,
                     "min_seq_len":5}

DNA_FILTER_CONFIG = {"min_rate":2,
                     "min_seq_len":7}

class Dataset(data.Dataset):
    """
    Nanopore DNA/RNA chunks dataset
    """
    def __init__(self, 
                 chunks:np.ndarray, 
                 seq:np.ndarray = None, 
                 seq_len:np.ndarray = None, 
                 chunks_len:np.ndarray = None,
                 transform:torchvision.transforms.transforms.Compose=None,
                 seq_padding:bool = True):
        """
        Generate a training dataset 

        Parameters
        ----------
        chunks : np.ndarray
            A N-by-M matrix, where N is the number of samples and M is the
            length of chunks.
        seq : np.ndarray, optional
            A N-by-D matrix, where N is the number of samples and D is the
            length of sequence. If this is given a seq_len vector must also
            be given to specify the length of the sequence. The default is None.
        seq_len : np.ndarray, optional
            A length N vector, gives the actual length of the sequence. 
            If not given then no label is used. The default is None.
        chunks_len: np.ndarray
            A length N vector, where N is the number of samples, indicate the
            length of the signal. If [M]*N vector will be used.
        transform : torchvision.transforms.transforms.Compose, optional
            DESCRIPTION. The default is None.
        seq_padding: bool, optional
            Default is True, if padding the sequence.

        Returns
        -------
        None.

        """        
        if seq_padding & (seq is not None):
            l_max = max(seq_len)
            seq = np.array([x+'$'*(l_max-len(x)) for x in seq])
        self.chunks = chunks[:,None,:].astype(np.float32)
        if chunks_len is None:
            self.chunks_len = np.array([[chunks.shape[1]]]*chunks.shape[0],dtype = np.int64)
        else:
            self.chunks_len = chunks_len[:,None].astype(np.int64)
        self.seq = seq
        if seq_len is not None:
            self.seq_len = seq_len[:,None].astype(np.int64)
        assert(not (seq is None)^(seq_len is None) )
        self.transform = transform
                
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.seq is None:
            sample = {'signal': self.chunks[idx],
                      'signal_len': self.chunks_len[idx]}
        else:
            sample = {'signal': self.chunks[idx], 
                      'signal_len': self.chunks_len[idx],
                      'seq': self.seq[idx],
                      'seq_len': self.seq_len[idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample

class NumIndex(object):
    """
    Convert string sequence to integer index array
    
    Parameters
    ----------
    alphabet_dict : Dict
        A dictionary map the string to index array.
            
    """
    def __init__(self,alphabet_dict:Dict):
        self.alphabet_dict = alphabet_dict
        self.alphabet_dict['$'] = 0 #Default padding character
    def __call__(self,sample:Dict):
        seq = list(sample['seq'])
        seq = [self.alphabet_dict[x] for x in seq]
        seq = np.array(seq,dtype = np.long)
        return {key:value if key != 'seq' else seq for key, value in sample.items()}

class ToTensor(object):
    def __call__(self,sample:Dict):
        func = torch.from_numpy
        return {key:func(value) for key, value in sample.items()}

def show_sample(sample:Dict, idx = 0):
    """
    Plot a sample signal, if input is a batch, plot the idx-th one.

    Parameters
    ----------
    sample : Dict
        A dictionary contain the samples.

    Returns
    -------
    None.

    """
    signal = sample['signal']
    if len(signal.shape) > 1:
        plt.plot(signal[idx,:])
    else:
        plt.plot(signal)

def filt(filt_config,chunks,seq,seq_len):
    segment_len = chunks.shape[1]
    print("Origin %d chunks in total."%(chunks.shape[0]))
    max_seq_len = np.int(segment_len/filt_config['min_rate'])
    mask = np.logical_and(seq_len>filt_config["min_seq_len"],seq_len<max_seq_len)
    print("%d chunks after filtering."%(sum(mask)))
    return chunks[mask],seq[mask],seq_len[mask]
    
def rna_filt(chunks,seq,seq_len):
    return partial(filt,RNA_FILTER_CONFIG)(chunks,seq,seq_len)

def dna_filt(chunks,seq,seq_len):
    return partial(filt,DNA_FILTER_CONFIG)(chunks,seq,seq_len)

if __name__ == "__main__":
    print("Load dataset.")
    chunks = np.load("/home/heavens/twilight_hdd1/m6A_Nanopore/6mA.zymo/191123.1.100pct/guppy_hac_extracted/chunks.npy")
    reference = np.load("/home/heavens/twilight_hdd1/m6A_Nanopore/6mA.zymo/191123.1.100pct/guppy_hac_extracted/seqs.npy")
    ref_len = np.load("/home/heavens/twilight_hdd1/m6A_Nanopore/6mA.zymo/191123.1.100pct/guppy_hac_extracted/seq_lens.npy")
    plt.hist(ref_len[ref_len<chunks.shape[1]],bins = 200)
    alphabet_dict = {'A':1,'C':2,'G':3,'T':4,'M':5}
    print("Filt dataset.")
    chunks,reference,ref_len = filt(RNA_FILTER_CONFIG,chunks,reference,ref_len)
    print("Create dataset.")
    dataset = Dataset(chunks,
                      seq = reference,
                      seq_len = ref_len,
                      transform = transforms.Compose([NumIndex(alphabet_dict),ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = 200,shuffle = True, num_workers = 4)
    for i_batch, sample_batch in enumerate(loader):
        if i_batch == 10:
            show_sample(sample_batch)
            break
    
