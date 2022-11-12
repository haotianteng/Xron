"""
@author: heavens
"""
from torch import nn
from torch import Tensor
import numpy as np
import torch
import xron.nn
import pandas as pd
from copy import deepcopy
from functools import partial
from itertools import permutations
from fast_ctc_decode import beam_search, viterbi_search
from numpy.random import default_rng
from xron.utils.vq import vq
from xron.utils.conformer import ConformerLayer
import editdistance
from typing import List,Dict

### Encoder Configuration
module_dict = {'Res1d':partial(xron.nn.Res1d,batch_norm = nn.LayerNorm),
               'Conformer':ConformerLayer,
               'BidirectionalRNN':xron.nn.BidirectionalRNN,
               'Attention_norm':xron.nn.AttentionNormalize,
               'LSTM':nn.LSTM,
               'ReLU':nn.ReLU,
               'GRU':nn.GRU,
               'Sigmoid':nn.Sigmoid}
# PORE_MODEL_F = "pore_models/5mer_level_table.model"
PORE_MODEL_F = "pore_models/m6A_5mer_level.model"
N_BASE = 5 #AGCTM
EMBEDDING_SIZE = 128
CNN_KERNAL_COMP= 25
CNN_STRIDE = 11
class CNN_CONFIG(object):
    CNN = {'N_Layer':3,
           'Layers': [{'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':16},
                      {'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32},
                      # {'layer_type':'Conformer','num_attention_heads':4,'ffn_dim':64,'depthwise_conv_kernel_size':7},
                      # {'layer_type':'Conformer','num_attention_heads':4,'ffn_dim':64,'depthwise_conv_kernel_size':7},
                      # {'layer_type':'Conformer','num_attention_heads':4,'ffn_dim':64,'depthwise_conv_kernel_size':7},
                      {'layer_type':'Res1d','kernel_size':CNN_KERNAL_COMP,'stride':CNN_STRIDE,'out_channels':768}]
        }
class RNN_CONFIG(CNN_CONFIG):
    RNN = {'layer_type':'BidirectionalRNN','hidden_size':768,'cell_type':'LSTM','num_layers':3}

class FNN_CONFIG(RNN_CONFIG):
    FNN = {'N_Layer':2,
           'Layers':[{'out_features':EMBEDDING_SIZE,'bias':True,'activation':'Sigmoid','dropout':None},
                     {'out_features':N_BASE+1,'bias':True,'activation':'Linear','dropout':None}]}
class CONFIG(FNN_CONFIG):
    def __init__(self):
        self.EMBEDDING= {'n_layers':self.CNN['N_Layer']+3}
        if self.EMBEDDING['n_layers'] ==  (self.CNN['N_Layer']+2):
            self.EMBEDDING['shape'] = EMBEDDING_SIZE*2
        else:
            self.EMBEDDING['shape'] = EMBEDDING_SIZE
        self.EMBEDDING['mask_cnn_ratio'] = 0.
            
### Critic Configuration
class CRITIC_CNN(object):
    CNN = {'N_Layer':3,
           'Layers': [{'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32},
                      {'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32},
                      {'layer_type':'Res1d','kernel_size':5,'stride':1,'out_channels':32}]
        }

class CRITIC_RNN(CRITIC_CNN):
    RNN = {'layer_type':'BidirectionalRNN','hidden_size':64,'cell_type':'LSTM','num_layers':1}
class CRITIC_FNN(CRITIC_RNN):
    FNN = {'N_Layer':2,
           'Layers':[{'out_features':32,'bias':True,'activation':'ReLU','dropout':0.2},
                     {'out_features':1,'bias':False,'activation':'Linear','dropout':None}]}
class CRITIC_CONFIG(CRITIC_FNN):
    pass


### Decoder configuration
module_dict['RevRes1d'] = xron.nn.RevRes1d

class DECODER_CONFIG(CONFIG):
    def __init__(self):
        super().__init__()
        self.CNN_DECODER = {'N_Layer':3,
                            'Layers': [{'layer_type':'RevRes1d','kernel_size':CNN_KERNAL_COMP,'stride':CNN_STRIDE,'out_channels':32},
                                       {'layer_type':'RevRes1d','kernel_size':5,'stride':1,'out_channels':32},
                                       {'layer_type':'RevRes1d','kernel_size':5,'stride':1,'out_channels':32}]
                            }
        self.CNN_DECODER['Input_Shape'] = self.EMBEDDING["shape"]
    FNN_DECODER = {'N_Layer':2,
                   'Layers':[{'out_features':8,'bias':True,'activation':'ReLU'},
                             {'out_features':1,'bias':True,'activation':'Linear'}]}
        
class MM_CONFIG(DECODER_CONFIG):
    PORE_MODEL = {"PORE_MODEL_F":PORE_MODEL_F,
                  "N_BASE": N_BASE, #for AGCT is 4, for AGCTM is 5
                  'K' : 3,
                  'N_EMBD':256, #Number of embbding, if None then it's set to N_BASE**K+1
                  "EMBEDDING_SIZE":EMBEDDING_SIZE,
                  "LOAD": False} #If load the pretrain pore model.
    DECODER = {"X_UPSAMPLING":5, #The scale factor of upsampling.
               "USE_STD":False}

def copy_config(config):
    class CONFIG:
        pass
    config_copy = CONFIG()
    config_modules = [x for x in config.__dir__() if not x .startswith('_')][::-1]
    for module in config_modules:
       setattr(config_copy,module,deepcopy(getattr(config,module)))
    return config_copy

class BASE(nn.Module):
    """The base class define the operation constructing a NN from configuration.
    """
    
    def _make_cnn(self,cnn_config:dict,in_channels = 1):
        cnn_config = deepcopy(cnn_config)
        layers = []
        for l in cnn_config['Layers']:
            layer_type = l.pop('layer_type')
            block = module_dict[layer_type]
            if layer_type == "Conformer":
                layers.append(xron.nn.Permute([0,2,1]))
            layers.append(block(in_channels = in_channels,**l))
            if layer_type == "Conformer":
                layers.append(xron.nn.Permute([0,2,1]))
            else:
                #For Conformer out_channels = in_channels
                in_channels = l['out_channels']
        return layers
    
    def _make_rnn(self,rnn_config,in_channels):
        rnn_config = deepcopy(rnn_config)
        block = module_dict[rnn_config.pop('layer_type')]
        cell = module_dict[rnn_config.pop('cell_type')]
        return [block(input_size = in_channels,
                            cell = cell,
                            **rnn_config)]
    
    def _make_fnn(self,fnn_config,in_channels):
        fnn_config = deepcopy(fnn_config)
        layers = []
        for l in fnn_config['Layers']:
            activation = l.pop('activation')
            if "dropout" in l:
                dropout = l.pop('dropout')
            else:
                dropout = None
            layers.append(nn.Linear(in_features = in_channels,**l))
            if activation != 'Linear':
                layers.append(module_dict[activation]())
            if dropout is not None:
                layers.append(nn.Dropout(p = dropout))
            in_channels = l['out_features']
        return layers
    
    def forward(self,batch):
        return self.net(batch)

class CRITIC(BASE):
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
        self.config = config
        cnn = self._make_cnn(config.CNN)
        permute = xron.nn.Permute([2,0,1]) #[N,C,L] -> [L,N,C]
        rnn = self._make_rnn(config.RNN,
                             in_channels = config.CNN['Layers'][-1]['out_channels'])
        directions = 2 if self.config.RNN['layer_type'] == "BidirectionalRNN" else 1
        fnn = self._make_fnn(config.FNN.copy(),
                             in_channels = config.RNN['hidden_size']*directions)
        self.net = nn.Sequential(*cnn,permute,*rnn,*fnn)
        self.mse_loss = nn.MSELoss(reduction = "none")
        


class CRNN(BASE):
    def __init__(self,config:CONFIG):
        """CRNN
        A Convolutional-Recurrent neural network for encoding the signal into
        base probability.

        Parameters
        ----------
        config : CONFIG
            The configuration used to generate the net, need to contain CNN,
            RNN and FNN attributes.
        """
        super().__init__()
        self.config = config
        # att = self._make_attention_norm(config.ATTENTION, 1)
        # cnn = self._make_cnn(config.CNN,in_channels = config.ATTENTION['Layers'][-1]['hidden_num'])
        cnn = self._make_cnn(config.CNN)
        permute = xron.nn.Permute([2,0,1]) #[N,C,L] -> [L,N,C]
        rnn = self._make_rnn(config.RNN,
                             in_channels = config.CNN['Layers'][-1]['out_channels'])
        directions = 2 if self.config.RNN['layer_type'] == "BidirectionalRNN" else 1
        fnn = self._make_fnn(config.FNN.copy(),
                             in_channels = config.RNN['hidden_size']*directions)
        log_softmax = nn.LogSoftmax(dim = 2)
        # self.net = nn.Sequential(*att,*cnn,permute,*rnn,*fnn,log_softmax)
        self.net = nn.Sequential(*cnn,permute,*rnn,*fnn,log_softmax)
        self.embedding_layers = [layer for layer in self.net[:self.config.EMBEDDING['n_layers']]]
        self.ctc = nn.CTCLoss(zero_infinity = False)
        self.nn_embd = vq
        if self.config.EMBEDDING["mask_cnn_ratio"] > 0:
            self.rng = default_rng()
    
    def forward_embedding(self,batch,embedding = None, training = True):
        for layer_i,layer in enumerate(self.embedding_layers):
            batch = layer(batch)
            if self.config.EMBEDDING["mask_cnn_ratio"] > 0 and layer_i == (self.config.CNN['N_Layer']-1) and training:
                T = batch.shape[-1]
                mask_T = int(T*self.config.EMBEDDING["mask_cnn_ratio"])
                mask_idxs = self.rng.choice(T, size = mask_T, replace = False)
                batch[:,:,mask_idxs] = 0 #Mask out channels
        if embedding:
            q = batch.permute([1,2,0]) # [L,N,C] -> [N,C,L]
            sg_q = q.detach()
            e,e_shadow = self.nn_embd(batch.permute([1,0,2]),embedding.weight) #[*,*,*] -> [N,L,C]
            e = e.permute([0,2,1]) #[N,L,C] -> [N,C,L]
            e_shadow = e_shadow.permute([0,2,1]) #[N,L,C] -> [N,C,L]
            sg_e = e.detach()
            return e,q,e_shadow,sg_q,sg_e
        else:
            return batch
    
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
                  beam_cut_threshold:float = 0.05,
                  reduction:Dict = None):
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
        reduction: dictionary, optional
            Default is None, conduct no reduction, if a dict is provided, the
            modifed bases will be reducted to original bases, e.g. {M:A} will
            reduct M base back to A and then compare the error.

        Returns
        -------
        The average error.

        """
        if torch.sum(torch.isnan(posteriors)):
            print("Posterior has NaN in it.")
            return np.nan
        preds = self.ctc_decode(posteriors = posteriors,
                                alphabet=alphabet,
                                beam_size=beam_size,
                                beam_cut_threshold=beam_cut_threshold)
        if reduction:
            error = self.reduction_error(preds, 
                                         seqs, 
                                         seqs_len,
                                         alphabet = alphabet[1:],
                                         modified_bases = reduction)
        else:
            error = self.error(preds,seqs,seqs_len,alphabet = alphabet[1:])
        return error
    
    def reduction_error(self,
              predictions:List[np.ndarray],
              seqs:Tensor,
              seqs_len:Tensor,
              modified_bases:Dict = {'M':'A'},
              alphabet:str = "ACGTM"):
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
            plain_seq = "".join([alphabet[x-1] for x in seqs[i][:seq_len]])
            for key,val in modified_bases.items():
                plain_seq.replace(key,val)
                palin_pred = pred.replace(key,val)
            error = editdistance.eval(palin_pred,plain_seq)
            errors.append(error/seq_len)
        return np.array(errors).mean()
    
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

class REVCNN(BASE):
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
        self.config = config
        self.embedding_size = config.CNN_DECODER['Input_Shape']
        cnn = self._make_cnn(config.CNN_DECODER, in_channels = self.embedding_size)
        permute = xron.nn.Permute([0,2,1]) #[N,C,L] -> [N,L,C]
        fnn = self._make_fnn(config.FNN_DECODER.copy(),
                             in_channels = config.CNN_DECODER['Layers'][-1]['out_channels'])
        self.net = nn.Sequential(*cnn,permute,*fnn)
        self.mse_loss = nn.MSELoss(reduction = 'none')
        self.entropy_loss = nn.CrossEntropyLoss()

class MM(nn.Module):
    def __init__(self,config:MM_CONFIG):
        """
        The statistical Markov generative pore model, 

        Parameters
        ----------
        config : DECODER_CONFIG
            A config contains the pore model.

        Returns
        -------
        None.
        """
        super().__init__()
        self.config = config
        self.N_BASE = self.config.PORE_MODEL['N_BASE']
        self.K = self.config.PORE_MODEL['K']
        self.n_embd = self.config.PORE_MODEL['N_EMBD']
        self.embedding_size = self.config.PORE_MODEL["EMBEDDING_SIZE"]
        if config.PORE_MODEL['LOAD']:
            self.pore_model = pd.read_csv(config.PORE_MODEL['PORE_MODEL_F'],
                                          delimiter = '\t')
            self.padding_signal = np.mean(self.pore_model.level_mean)
            level_tensor = torch.tensor([self.padding_signal]+list(self.pore_model.level_mean)).unsqueeze(1).to(torch.float32)
            self.level_embedding = nn.Embedding.from_pretrained(level_tensor,
                                                                freeze = False,
                                                                padding_idx=0)
        else:
            if not self.n_embd:
                n_embd = (self.N_BASE)**self.K+1
            else:
                n_embd = self.n_embd
            self.level_embedding = nn.Embedding(n_embd,self.embedding_size)
            # if self.config.EMBEDDING['n_layers'] ==  (self.config.CNN['N_Layer']+4):
            #     shift = 0.5 #The embedding output is from a sigmoid layer, so it's approximately sigmoid(0) ~ 0.5
            # else:
            #     shift = 0.
            self.level_embedding.weight.data.uniform_(-1./n_embd, 1./n_embd)
        self.upsampling = torch.nn.Upsample(scale_factor=config.DECODER['X_UPSAMPLING'],
                                     mode = 'nearest')
        self.entropy_loss = nn.CrossEntropyLoss()
        self.bn = torch.nn.BatchNorm1d(1)
        
    def forward(self, sampling:torch.Tensor,device = None):
        sampling = sampling.cpu().detach().numpy()
        sampling = np.argmax(sampling,axis = 1)
        kmer_batch = self._kmer_decode(sampling)
        if device:
            kmer_batch = kmer_batch.to(device = device)
        rc_signal = self.level_embedding(kmer_batch).squeeze(2).unsqueeze(1) # (N,L,1) -> (N,1,L)
        rc_signal = self.bn(self.upsampling(rc_signal))
        return rc_signal.permute([0,2,1])
        
    def _kmer_decode(self,
                     sequence_batch):
        kmer_batch = np.zeros((sequence_batch.shape[0],sequence_batch.shape[1]))
        K = self.K
        for i,sequence in enumerate(sequence_batch):
            kmer_seq = []
            curr_kmer = ''
            pre_base = 0
            for base in sequence:
                if base == 0 or base == pre_base:
                    curr_kmer = curr_kmer
                else:
                    curr_kmer = (curr_kmer + str(base))[-K:]
                pre_base = base
                kmer_seq.append(self.kmer2idx(curr_kmer))
            kmer_seq = np.asarray(kmer_seq)
            kmer_batch[i,:] = kmer_seq+1
        # signal_batch = torch.from_numpy(signal_batch[:,None,:])
        if kmer_batch.max() >= self.level_embedding.weight.shape[0]:
            print(kmer_batch.max())
            print(self.level_embedding.weight.shape)
        return torch.LongTensor(kmer_batch)
    
    def kmer2idx(self,kmer:str):
        "Map the kmer to index, e.g. AAAAA -> 0, AAAAC -> 1, AAAAG -> 2, ..."
        if len(kmer)<self.K:
            return -1
        multi = 1
        idx = 0
        for base in kmer[::-1]:
            idx += (int(base)-1)*multi
            multi = multi * self.N_BASE
        return idx
    
    def idx2kmer(self,idx:int,ab_dict = None):
        if not ab_dict:
            ab_dict = {i:str(i+1) for i in np.arange(self.N_BASE)}
        kmer = ''
        while idx>=self.N_BASE:
            kmer = ab_dict[idx%self.N_BASE] + kmer
            idx //= self.N_BASE
        kmer = ab_dict[idx%self.N_BASE] + kmer
        kmer = ab_dict[0]*(self.K-len(kmer)) + kmer              
        return kmer
            
    @property
    def mse_loss(self):
        if self.config.DECODER['USE_STD']:
            pass #TODO: Implement the std version.
        else:
            return nn.MSELoss(reduction= 'none')
    
    def save_embedding(self,alphabeta = 'ACGTM'):
        embedding = self.level_embedding.weight.detach().cpu().squeeze(1).numpy()
        ab_dict = {i:x for i,x in enumerate(alphabeta)}
        with open(self.config.PORE_MODEL['PORE_MODEL_F'],'w+') as f:
            f.write("kmer\tlevel_mean\n")
            for i,e in enumerate(embedding[1:]):
                f.write("%s\t%.6f\n"%(self.idx2kmer(i,ab_dict = ab_dict),e))
    
if __name__ == "__main__":
    config = CONFIG()
    encoder = CRNN(config)
    batch_size = 88
    chunk_len = 2000
    test_batch = torch.randn(batch_size,1,chunk_len)
    embedding = encoder.forward_embedding(test_batch)
    print("Input: ",test_batch.shape,"Encoded: ",embedding.shape)
    decoder_config = DECODER_CONFIG()
    decoder = REVCNN(decoder_config)
    rc = decoder.forward(embedding.permute([1,2,0])).permute([0,2,1])
    print("Reconstructed:",rc.shape)
    # output = encoder.forward(test_batch)
    # mm_config = MM_CONFIG()
    # mm_decoder = MM(mm_config)
    # mm_rc = mm_decoder.forward(output.permute([1,2,0]))
    # print("MM Reconstructed:",mm_rc.shape)
    # mm_decoder.save_embedding()
