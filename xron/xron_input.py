"""
Created on Thu Mar  4 19:33:04 2021

@author: Haotian Teng
"""
import numpy as np
import lmdb
import torchvision
import torch
import pickle
from torch import nn
import torch.utils.data as data
from functools import lru_cache
from torchvision import transforms
from matplotlib import pyplot as plt
from typing import Dict,Callable

def seq_collate_fn(batch):
    """
    Collate function for the dataset, which will pad the signal and sequence.

    Parameters
    ----------
    batch : List
        A list of samples.

    Returns
    -------
    Dict
        A dictionary contain the batch data.

    """
    signal = [x['signal'] for x in batch]
    signal = torch.stack(signal)
    signal_len = [x['signal_len'] for x in batch]
    signal_len = torch.stack(signal_len)
    sample = {'signal': signal, 'signal_len': signal_len}
    if 'seq' in batch[0]:
        seq = [x['seq'] for x in batch]
        seq_len = [x['seq_len'] for x in batch]
        seq_len = torch.stack(seq_len)
        seq = nn.utils.rnn.pad_sequence(seq,batch_first = True, padding_value = 0)
        sample['seq'] = seq
        sample['seq_len'] = seq_len
    return sample

class LMDBDataset(data.Dataset):
    """
    Nanopore DNA/RNA chunks dataset in LMDB format
    """
    def __init__(self,
                 lmdb_path:str,
                 transform:torchvision.transforms.transforms.Compose=None,
                 ):
        """
        Generate a training dataset 

        Parameters
        ----------
        lmdb_path : str
            The path of the lmdb file.
        transform : torchvision.transforms.transforms.Compose, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """        
        self.lmdb_env = lmdb.open(lmdb_path, 
                                  readonly=True,
                                  lock = False,
                                  )
        self.transform = transform
    
    @lru_cache(maxsize=1)
    def __len__(self):
        return self.lmdb_env.stat()['entries']

    @lru_cache(maxsize=1000)
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write = False) as txn:
            sample_buffer = None
            while sample_buffer is None:
                sample_buffer = txn.get(str(idx).encode())
                idx = (idx + 1) % self.__len__()
            sample_dict = pickle.loads(sample_buffer)
            sample = {}
            sample['signal'] = np.frombuffer(sample_dict['signal'],dtype = np.float32)[None,:]
            sample['signal_len'] = np.asarray([sample_dict['signal_len']],dtype = np.int64)
            if 'seq' in sample_dict:
                sample['seq'] = sample_dict['seq']
                sample['seq_len'] = np.asarray([sample_dict['seq_len']], dtype = np.int64)
            if self.transform:
                sample = self.transform(sample)
            return sample

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
        self.chunks = chunks
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
            sample = {'signal': self.chunks[idx][None,:].astype(np.float32),
                      'signal_len': self.chunks_len[idx]}
        else:
            sample = {'signal': self.chunks[idx][None,:].astype(np.float32), 
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
        self.alphabet_dict['N'] = 0 #Remove unknonw nucleotide
    def __call__(self,sample:Dict):
        seq = list(sample['seq'])
        seq = [self.alphabet_dict[x] for x in seq]
        seq = np.array(seq,dtype = np.int64)
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
    fig = plt.figure()
    signal = sample['signal']
    if len(signal.shape) > 1:
        plt.plot(signal[idx][0].numpy())
    else:
        plt.plot(signal.numpy())
    plt.show()

def check_sample(sample:Dict):
    seq_batch = sample['seq']
    seq_len_batch = sample['seq_len']
    signal_batch = sample['signal']
    for i in range(seq_batch.shape[0]):
        seq = seq_batch[i]
        seq_len = seq_len_batch[i]
        if not all(seq[:seq_len].numpy()!=0):
            print("Sequence with invalid neucleotide found.")
        signal = signal_batch[i]
        #check if nan is in signal
        if torch.any(torch.isnan(signal)):
            print("Nan signal found.")
        if torch.all(signal == 0):
            print("Zero signal found.")
        signal_len = sample['signal_len'][i]
        #check if signal length is correct
        # assert(signal_len >0)

if __name__ == "__main__":
    print("Load dataset.")
    folder = "/data/HEK293T_RNA004/extracted/"
    chunks = np.load(folder + "chunks.npy")
    reference = np.load(folder + "seqs.npy")
    ref_len = np.load(folder + "seq_lens.npy")
    plt.hist(ref_len[ref_len<chunks.shape[1]],bins = 200)
    alphabet_dict = {'A':1,'C':2,'G':3,'T':4,'M':5}
    print("Filt dataset.")
    # chunks,reference,ref_len = filt(RNA_FILTER_CONFIG,chunks,reference,ref_len)
    print("Create dataset.")
    dataset = Dataset(chunks,
                      seq = reference,
                      seq_len = ref_len,
                      transform = transforms.Compose([NumIndex(alphabet_dict),ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = 200,shuffle = True, num_workers = 4)
    for i_batch, sample_batch in enumerate(loader):
        check_sample(sample_batch)
        if i_batch == 1000:
            print(sample_batch['signal'].shape)
            print(sample_batch['signal_len'].shape)
            print(sample_batch['seq'].shape)
            print(sample_batch['seq_len'].shape)
            print(sample_batch['signal'].dtype)
            print(sample_batch['signal_len'].dtype)
            print(sample_batch['seq'].dtype)
            print(sample_batch['seq_len'].dtype)
            show_sample(sample_batch)
            break
    #%% Testing LMDB dataset
    print("Load LMDB dataset.")
    LMDB_folder = "/data/HEK293T_RNA004/extracted/"
    alphabet_dict = {'A':1,'C':2,'G':3,'T':4,'M':5}
    LMDB_dataset = LMDBDataset(LMDB_folder,
                               transform = transforms.Compose([NumIndex(alphabet_dict),ToTensor()]))
    eval_size = 1000
    dataset,eval_ds = torch.utils.data.random_split(LMDB_dataset,[len(LMDB_dataset) - eval_size, eval_size], generator=torch.Generator().manual_seed(42))
    dataloader = data.DataLoader(dataset,batch_size = 200,shuffle = True, num_workers = 4,collate_fn = seq_collate_fn)
    eval_dataloader = data.DataLoader(eval_ds,batch_size = 200,shuffle = True, num_workers = 0, collate_fn=seq_collate_fn)
    zero_signal = 0
    n_epoches = 5
    from tqdm import tqdm
    for epoch_i in range(n_epoches):
        for i_batch, sample_batch in enumerate(tqdm(dataloader)):
            if i_batch> 100:
                break
            # check_sample(sample_batch)
            # signal_len = sample_batch['signal_len']
            # zero_signal += torch.sum(signal_len == 0).item()
            # if i_batch == 100:
            #     print(sample_batch['signal'].shape)
            #     print(sample_batch['signal_len'].shape)
            #     print(sample_batch['seq'].shape)
            #     print(sample_batch['seq_len'].shape) 
            #     print(sample_batch['signal'].dtype)
            #     print(sample_batch['signal_len'].dtype)
            #     print(sample_batch['seq'].dtype)
            #     print(sample_batch['seq_len'].dtype)
            #     show_sample(sample_batch)
            #     break
        for i_batch, sample_batch in enumerate(tqdm(eval_dataloader)):
            pass

