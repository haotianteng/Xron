"""
@author: heavens
"""
from torch import nn
from torch import Tensor
import numpy as np
import torch
import xron.nn
from copy import deepcopy
from functools import partial
from itertools import permutations
from fast_ctc_decode import beam_search, viterbi_search
import editdistance
from typing import List

### Encoder Configuration
module_dict = {'Res1d':xron.nn.Res1d,
               'BidirectionalRNN':xron.nn.BidirectionalRNN,
               'LSTM':nn.LSTM,
               'ReLU':nn.ReLU,
               'Sigmoid':nn.Sigmoid}
class CNN_CONFIG(object):
    CNN = {'N_Layer':3,
           'Layers': [{'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32},
                      {'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32},
                      {'layer_type':'Res1d','kernel_size':15,'stride':5,'out_channels':64}]
        }

class RNN_CONFIG(CNN_CONFIG):
    RNN = {'layer_type':'BidirectionalRNN','hidden_size':64,'cell_type':'LSTM','num_layers':3}

class FNN_CONFIG(RNN_CONFIG):
    FNN = {'N_Layer':2,
           'Layers':[{'out_features':32,'bias':True,'activation':'ReLU'},
                     {'out_features':5,'bias':False,'activation':'Linear'}]}
    
class CONFIG(FNN_CONFIG):
    pass

### Decoder configuration
module_dict['RevRes1d'] = xron.nn.RevRes1d

class DECODER_CONFIG(CONFIG):
    CNN_DECODER = {'N_Layer':3,
              'Layers': [{'layer_type':'RevRes1d','kernel_size':15,'stride':5,'out_channels':32},
                         {'layer_type':'RevRes1d','kernel_size':5,'stride':1,'out_channels':32},
                         {'layer_type':'RevRes1d','kernel_size':5,'stride':1,'out_channels':32}]
             }
    FNN_DECODER = {'N_Layer':1,
                   'Layers':[{'out_features':8,'bias':True,'activation':'ReLU'},
                             {'out_features':1,'bias':False,'activation':'Linear'}]}
            
class CRNN(nn.Module):
    def __init__(self,config:CONFIG):
        """
        A Convolutional-Recurrent neural network for encoding the signal into
        base probability.

        Parameters
        ----------
        config : CONFIG
            The configuration used to generate the net, need to contain CNN,
            RNN and FNN attributes.
        """
        super().__init__()
        self.config = self._copy_config(config)
        config = self._copy_config(self.config)
        cnn = self._make_cnn(config.CNN)
        permute = xron.nn.Permute([2,0,1]) #[N,C,L] -> [L,N,C]
        rnn = self._make_rnn(config.RNN,
                             in_channels = config.CNN['Layers'][-1]['out_channels'])
        directions = 2 if self.config.RNN['layer_type'] == "BidirectionalRNN" else 1
        fnn = self._make_fnn(config.FNN.copy(),
                             in_channels = config.RNN['hidden_size']*directions)
        log_softmax = nn.LogSoftmax(dim = 2)
        self.net = nn.Sequential(*cnn,permute,*rnn,*fnn,log_softmax)
        self.ctc = nn.CTCLoss()
    
    def _copy_config(self,config):
        class CONFIG:
            pass
        config_copy = CONFIG()
        config_modules = [x for x in config.__dir__() if not x .startswith('_')][::-1]
        for module in config_modules:
            setattr(config_copy,module,deepcopy(getattr(config,module)))
        return config_copy
    
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
            if activation != 'Linear':
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
    
    def ctc_decode(self,
                   posteriors:Tensor,
                   alphabet:str = "NACGT",
                   beam_size:int = 5,
                   beam_cut_threshold:float = 0.05):
        """
        Use beam-search CTC decoder to decode the given posterior probability.

        Parameters
        ----------
        posteriors : Tensor
            A L-N-C posterior probability tensor. Where L is the signal length,
            N is the batch size and C is the number of alphabet.
        alphabet : str, optional
            A string contains the alphabet, first character is the blank symbol.
            The default is "NACGT".
        beam_size : int, optional
            The beam size, if it's 1, a faster but less-accurate vaterbi
            decoder will be used. The default is 5.
        beam_cut_threshold: float, optional
            The cut threshold of beam search, the higher it is, more search
            will done, too large the value could run out of search space of 
            beam search. The default is 0.05

        Returns
        -------
        A N-length list of the predicting sequence.

        """
        if beam_size <=1:
            decoder = partial(viterbi_search, alphabet = alphabet)
        else:
            decoder = partial(beam_search, beam_size = beam_size, alphabet = alphabet)
        posteriors = posteriors.cpu().detach().numpy()
        batch_size = posteriors.shape[1]
        predictions = []
        for i in np.arange(batch_size):
            pred = decoder(np.exp(posteriors[:,i,:]))[0]
            predictions.append(pred)
        return predictions
        
    
    def error(self,
              predictions:List[np.ndarray],
              seqs:Tensor,
              seqs_len:Tensor,
              alphabet:str = "ACGT"):
        """
        Calculate the ctc decoding error between the true label.

        Parameters
        ----------
        predictions: List[np.ndarray]
            A list of vectors, each vector is the predicting sequence.
        seqs : Tensor
            A N-L matrix contains the true sequence label.
        seqs_len : Tensor
            A N-1 vector contains the length of the sequences.
        alphabet : str, optional
            A string contains the alphabet without the blank symbol.
            Can be a permutation of the alphabet used in ctc_decode.
            The default is "ACGT".

        Returns
        -------
        None.

        """
        batch_size = len(predictions)
        seqs = seqs.cpu().detach().numpy()
        seqs_len = seqs_len.cpu().detach().numpy().flatten()
        errors = []
        for i in np.arange(batch_size):
            pred = predictions[i]
            seq_len = seqs_len[i]
            error = editdistance.eval(pred,"".join([alphabet[x-1] for x in seqs[i][:seq_len]]))
            errors.append(error/seq_len)
        return np.array(errors).mean()
    
    def ctc_error(self,
                  posteriors:Tensor,
                  seqs:Tensor,
                  seqs_len:Tensor,
                  alphabet:str = "NACGT",
                  beam_size:int = 5,
                  beam_cut_threshold:float = 0.05):
        """
        Calcualte the CTC decoding error of the given posterior probability
        and sequence with true label.

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
        beam_cut_threshold: float, optional
            The cut threshold of beam search, the higher it is, more search
            will done, too large the value could run out of search space of 
            beam search. The default is 0.05

        Returns
        -------
        The average error.

        """
        preds = self.ctc_decode(posteriors = posteriors,
                                alphabet=alphabet,
                                beam_size=beam_size,
                                beam_cut_threshold=beam_cut_threshold)
        error = self.error(preds,seqs,seqs_len,alphabet = alphabet[1:])
        return error
    
    def permute_error(self,
                      posteriors:Tensor,
                      seqs:Tensor,
                      seqs_len:Tensor,
                      alphabet:str = "NACGT",
                      beam_size:int = 5,
                      beam_cut_threshold:float = 0.05):
        """
        Calcualte the permutating error of the given posterior probability
        and sequence with true label. Permutation of the alphabet will be used
        to calculate the minimum error given the CTC decoding result.

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
            The default is "NACGT". All permutations of the alphabet will be tested.
        beam_size : int, optional
            The beam size, if it's 1, a faster but less-accurate vaterbi
            decoder will be used. The default is 5.
        beam_cut_threshold: float, optional
            The cut threshold of beam search, the higher it is, more search
            will done, too large the value could run out of search space of 
            beam search. The default is 0.05

        Returns
        -------
        A list of errors with the permutation.

        """
        preds = self.ctc_decode(posteriors = posteriors,
                                alphabet=alphabet,
                                beam_size=beam_size,
                                beam_cut_threshold=beam_cut_threshold)
        ab = alphabet[1:]
        errors = []
        perms = []
        for perm in permutations(ab):
            perm = ''.join(perm)
            error = self.error(preds,seqs,seqs_len,alphabet = perm)
            errors.append(error)
            perms.append(perm)
        return np.asarray(errors), np.asarray(perms)

class REVCNN(nn.Module):
    def __init__(self,config:DECODER_CONFIG):
        """
        The reverse convolutional neural network to reconstruct the signal from
        the base probability, the decoder.

        Parameters
        ----------
        config : DECODER_CONFIG
            A Configuration used to generate this net, need to contain 
            CNN_DECODER and FNN_DECODER attributes. Input should be a tensor 
            with shape [N,C,L], output is a tensor with shape [N,L,C]

        Returns
        -------
        None.

        """
        super().__init__()
        self.config = self._copy_config(config)
        self.n_base = config.FNN['Layers'][-1]['out_features']
        config = self._copy_config(self.config)
        cnn = self._make_cnn(config.CNN_DECODER, in_channels = self.n_base)
        permute = xron.nn.Permute([0,2,1]) #[N,C,L] -> [N,L,C]
        fnn = self._make_fnn(config.FNN_DECODER.copy(),
                             in_channels = config.CNN_DECODER['Layers'][-1]['out_channels'])
        self.net = nn.Sequential(*cnn,permute,*fnn)
        self.mse_loss = nn.MSELoss(reduction = 'mean')
        self.entropy_loss = nn.CrossEntropyLoss()
    
    def _copy_config(self,config):
        class CONFIG:
            pass
        config_copy = CONFIG()
        config_modules = [x for x in config.__dir__() if not x .startswith('_')][::-1]
        for module in config_modules:
            setattr(config_copy,module,deepcopy(getattr(config,module)))
        return config_copy
    
    def _make_cnn(self,cnn_config,in_channels = 1):
        layers = []
        for l in cnn_config['Layers']:
            block = module_dict[l.pop('layer_type')]
            layers.append(block(in_channels = in_channels,**l))
            in_channels = l['out_channels']
        return layers
    
    def _make_fnn(self,fnn_config,in_channels):
        layers = []
        for l in fnn_config['Layers']:
            activation = l.pop('activation')
            layers.append(nn.Linear(in_features = in_channels,**l))
            if activation != 'Linear':
                layers.append(module_dict[activation]())
            in_channels = l['out_features']
        return layers
        
    def forward(self,batch):
        return self.net(batch)

if __name__ == "__main__":
    config = CONFIG()
    encoder = CRNN(config)
    batch_size = 88
    chunk_len = 2000
    test_batch = torch.randn(batch_size,1,chunk_len)
    output = encoder.forward(test_batch)
    print("Input: ",test_batch.shape,"Encoded: ",output.shape)
    decoder_config = DECODER_CONFIG()
    decoder = REVCNN(decoder_config)
    rc = decoder.forward(output.permute([1,2,0])).permute([0,2,1])
    print("Reconstructed:",rc.shape)