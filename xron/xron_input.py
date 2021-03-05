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
from typing import Dict
class Dataset(data.Dataset):
    """
    Nanopore DNA/RNA chunks dataset
    """
    def __init__(self, 
                 chunks:np.ndarray, 
                 seq:np.ndarray = None, 
                 seq_len:np.ndarray = None, 
                 chunks_len:np.ndarray = None,
                 transform:torchvision.transforms.transforms.Compose=None):
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

        Returns
        -------
        None.

        """        
        self.chunks = chunks[:,None,:].astype(np.float32)
        if chunks_len is None:
            self.chunks_len = np.array([[chunks.shape[1]]]*chunks.shape[0],dtype = np.int16)
        else:
            self.chunks_len = chunks_len[:,None]
        self.seq = seq
        if seq_len is not None:
            self.seq_len = seq_len[:,None].astype(np.int16)
        else:
            self.seq_len = seq_len
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
    
if __name__ == "__main__":
    chunks = np.load("/home/heavens/twilight_data1/S10_DNA/20170322_c4_watermanag_S10/bonito_output/ctc_data/chunks.npy")
    reference = np.load("/home/heavens/twilight_data1/S10_DNA/20170322_c4_watermanag_S10/bonito_output/ctc_data/references.npy")
    ref_len = np.load("/home/heavens/twilight_data1/S10_DNA/20170322_c4_watermanag_S10/bonito_output/ctc_data/reference_lengths.npy")
    dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = 200,shuffle = True, num_workers = 4)
    for i_batch, sample_batch in enumerate(loader):
        if i_batch == 10:
            show_sample(sample_batch)
            break
    