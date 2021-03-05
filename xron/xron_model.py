"""
Created on Thu Feb 18 21:21:51 2021

@author: heavens
"""
from torch import nn
from torch import Tensor
import numpy as np
import torch
import xron.nn
from typing import Dict
from functools import partial
from fast_ctc_decode import beam_search, viterbi_search
import editdistance
class CNN_CONFIG(object):
    CNN = {'N_Layer':3,
           'Layers': [{'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32},
                      {'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32},
                      {'layer_type':'Res1d','kernel_size':15,'stride':5,'out_channels':64}]
        }

class RNN_CONFIG(CNN_CONFIG):
    RNN = {'layer_type':'BidirectionalRNN','hidden_size':64,'cell_type':'LSTM','num_layers':3}
    FNN = {'N_Layer':2,
           'Layers':[{'out_features':32,'bias':True,'activation':'ReLU'},
                     {'out_features':6,'bias':False,'activation':None}]}
    
class CONFIG(RNN_CONFIG):
    pass

module_dict = {'Res1d':xron.nn.Res1d,
               'BidirectionalRNN':xron.nn.BidirectionalRNN,
               'LSTM':nn.LSTM,
               'ReLU':nn.ReLU,
               'Sigmoid':nn.Sigmoid}
        
        
class CRNN(nn.Module):
    def __init__(self,config:Dict):
        super().__init__()
        cnn = self._make_cnn(config.CNN.copy())
        permute = xron.nn.Permute([2,0,1]) #[N,C,L] -> [L,N,C]
        rnn = self._make_rnn(config.RNN.copy(),
                             in_channels = config.CNN['Layers'][-1]['out_channels'])
        directions = 2 if config.RNN['layer_type'] == "BidirectionalRNN" else 1
        fnn = self._make_fnn(config.FNN.copy(),
                             in_channels = config.RNN['hidden_size']*directions)
        log_softmax = nn.LogSoftmax()
        self.net = nn.Sequential(*cnn,permute,*rnn,*fnn,log_softmax)
        self.ctc = nn.CTCLoss()
    
    def _make_cnn(self,cnn_config,in_channels = 1):
        layers = []
        for l in cnn_config['Layers']:
            block = module_dict[l.pop('layer_type')]
            layers.append(block(in_channels = in_channels,**l))
            in_channels = l['out_channels']
        return layers
    
    def _make_rnn(self,rnn_config,in_channels):
        block = module_dict[rnn_config.pop('layer_type')]
        cell = module_dict[rnn_config.pop('cell_type')]
        return [block(input_size = in_channels,
                            cell = cell,
                            **rnn_config)]
    
    def _make_fnn(self,fnn_config,in_channels):
        layers = []
        for l in fnn_config['Layers']:
            activation = l.pop('activation')
            layers.append(nn.Linear(in_features = in_channels,**l))
            if activation is not None:
                layers.append(module_dict[activation]())
            in_channels = l['out_features']
        return layers
        
    def forward(self,batch):
        return self.net(batch)
    
    def ctc_loss(self,
                 posterior:Tensor,
                 posterior_len:Tensor,
                 seq:Tensor,
                 seq_len:Tensor):
        return self.ctc(posterior,seq,posterior_len,seq_len)
    
    def ctc_error(self,
                  posteriors:Tensor,
                  seqs:Tensor,
                  seqs_len:Tensor,
                  alphabet:str = "NACGT",
                  beam_size:int = 5):
        """
        Calculate the ctc decoding error between the true label.

        Parameters
        ----------
        posteriors : Tensor
            A L-N-C posterior probability tensor. Where L is the signal length,
            N is the batch size and C is the number of alphabet.
        seqs : Tensor
            A N-L matrix contains the true sequence label.
        seqs_len : Tensor
            A N-1 vector contains the length of the sequences.
        alphabet : str, optional
            A string contains the alphabet, first character is the blank symbol.
            The default is "NACGT".
        beam_size : int, optional
            The beam size, if it's 1, a faster but less-accurate vaterbi
            decoder will be used. The default is 5.

        Returns
        -------
        None.

        """
        if beam_size <=1:
            decoder = viterbi_search
        else:
            decoder = partial(beam_search, beam_size = beam_size)
        posteriors = output.numpy()
        batch_size = posteriors.shape[0]
        seqs = seqs.numpy().copy()
        seqs_len = seqs_len.numpy().copy().flatten()
        errors = []
        ab = alphabet[1:]
        for i in np.arange(batch_size):
            pred = decoder(posteriors[:,i,:])
            seq_len = seqs_len[i]
            error = editdistance(pred,"".join([ab[x] for x in seqs[i][:seq_len]]))
            errors.append(error/seq_len)
        return np.array(errors).mean()

if __name__ == "__main__":
    config = CONFIG()
    net = CRNN(config)
    batch_size = 88
    chunk_len = 2000
    test_batch = torch.randn(batch_size,1,chunk_len)
    output = net.forward(test_batch)
    print("Input",test_batch.shape,"Output",output.shape)