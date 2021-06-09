"""
@author: Haotian Teng
"""
import os 
import sys
import torch
import argparse
import numpy as np
from typing import Union,List
from itertools import chain
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from xron.xron_input import Dataset, ToTensor
from xron.xron_train_base import Trainer, DeviceDataLoader, load_config
from xron.xron_model import REVCNN,DECODER_CONFIG,CRNN,MM_CONFIG,MM
from xron.xron_label import MetricAligner
from torch.distributions.one_hot_categorical import OneHotCategorical as OHC

class VAETrainer(Trainer):
    def __init__(self,
                 train_dataloader:DataLoader,
                 encoder:CRNN,
                 decoder:REVCNN,
                 mm:MM,
                 config:Union[DECODER_CONFIG,MM_CONFIG],
                 aligner:MetricAligner,
                 device:str = None,
                 eval_dataloader:DataLoader = None):
        """

        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader.
        encoder: CRNN
            A Convolutional-Recurrent Neural Network
        decoder : REVCNN
            REVCNN decoder
        mm: MM
            Markov Model instance.
        device: str
            The device used to train the model, can be 'cpu' or 'cuda'.
            Default is None, use cuda device if it's available.
        config: Union[DECODER_CONFIG,MM_CONFIG]
            A CONFIG class contains unsupervised training configurations. Need 
            to contain at least these parameters: keep_record, device and 
            grad_norm.
        eval_dataloader : DataLoader, optional
            Evaluation dataloader, if None training dataloader will be used.
            The default is None.

        """
        super().__init__(train_dataloader=train_dataloader,
                         nets = {"encoder":encoder,
                                 "decoder":decoder,
                                 "mm":mm},
                         config = config,
                         device = device,
                         eval_dataloader = eval_dataloader)
        self.encoder = encoder
        self.decoder = decoder
        self.mm = mm
        self.aligner = aligner
        params = [encoder.parameters(),decoder.parameters(),mm.parameters()]
        d_params = [decoder.parameters(),mm.parameters()]
        self.decoder_parameters = chain(*d_params)
        self.encoder_parameters = encoder.parameters()
        self.parameters = chain(*params)
        self.train_config = config.TRAIN
        
        self.decay = self.train_config["decay"]
        self.global_step = 0
        self.awake = False
        self.score_average = 0
        self.records = {'rc_losses':[],
                        'rc_losses_encoder':[],
                        'entropy_losses':[],
                        'alignment_score':[],
                        'valid_alignment':[]}
    def train(self,
              epoches:int,
              optimizers:List[torch.optim.Optimizer],
              save_cycle:int,
              save_folder:str):
        """
        Train the encoder-decodr nets.

        Parameters
        ----------
        epoches : int
            Number of epoches to train.
        optimizers : List[torch.optim.Optimizer]
            A list of two optimizers, the first one is optimizer training the
            encoder parameters and the second one for decoder parameters.
        save_cycle : int
            Save every save_cycle batches.
        save_folder : str
            The folder to save the model and training records.

        Returns
        -------
        None.

        """
        self.save_folder = save_folder
        self._save_config()
        records = self.records
        preheat = self.train_config["preheat"]
        e_decay = self.train_config["epsilon_decay"]
        epsilon = self.train_config["epsilon"]
        alpha = self.train_config["alpha"]
        beta = self.train_config["beta"]
        gamma = self.train_config["gamma"]
        opt_e,opt_g,opt_d = optimizers
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                if self.global_step < preheat:
                    self.epsilon = 1
                    use_revcnn = False
                else:
                    self.epsilon = max(1+(self.global_step - preheat)*e_decay,epsilon)
                    for param in self.mm.parameters():
                        param.requires_grad = False
                    use_revcnn = True
                losses = self.train_step(batch)
                loss = -alpha*losses[0]+beta*losses[1]+gamma*losses[2] #Maximize -H(q), minimize -(cross entropy loss)
                opt_e.zero_grad()
                loss.backward()
                _,_,_,loss = self.train_step(batch,comp_signal = use_revcnn)
                if use_revcnn:
                    opt_d.zero_grad()
                    loss.backward()
                    opt_d.step()
                else:
                    opt_g.zero_grad()
                    loss.backward()
                    opt_g.step()
                opt_e.step()
                if (self.global_step+1)%save_cycle==0:
                    self.save()
                    eval_i,valid_batch = next(enumerate(self.eval_ds))
                    valid_error,valid_perm = self.valid_step(valid_batch)
                    records['entropy_losses'].append(losses[0].detach().cpu().numpy()[()])
                    records['valid_alignment'].append(valid_error)
                    print("Epoch %d Batch %d, entropy_loss %f, rc_loss %f, encod valid_error %f, perm:%s, alignment_loss %f"%(epoch_i, i_batch, losses[0], losses[3], valid_error, valid_perm, losses[2]))
                    records['alignment_score'].append(losses[2].detach().cpu().numpy()[()])
                    records['rc_losses_encoder'].append(losses[1].detach().cpu().numpy()[()])
                    self._update_records()
                losses = None
                torch.nn.utils.clip_grad_norm_(self.parameters, 
                                               max_norm=self.grad_norm)
                self.global_step +=1
        
    def train_step(self,batch,comp_signal = False):
        encoder = self.encoder
        decoder = self.decoder
        mm = self.mm
        signal = batch['signal']
        logprob = encoder.forward(signal) #[L,N,C]
        if np.random.rand()<self.epsilon:
            m = OHC(logits = logprob)
            sampling = m.sample().permute([1,2,0]) #[L,N,C]->[N,C,L]
        else:
            sampling = torch.argmax(logprob,dim = 2)
            sampling = torch.nn.functional.one_hot(sampling,num_classes = logprob.shape[2]).permute([1,2,0]).float()
        rc_mm = mm.forward(sampling,device = self.device).permute([0,2,1]) #[N,L,C] -> [N,C,L]
        rc_revcnn = decoder.forward(sampling).permute([0,2,1]) #[N,L,C] -> [N,C,L]
        rc_signal = rc_mm
        if comp_signal:
            rc_signal += rc_revcnn
        mse_loss = decoder.mse_loss(rc_signal,signal)
        entropy_loss = decoder.entropy_loss(logprob.permute([1,2,0]),sampling.max(dim = 1)[1])
        raw_seq = torch.argmax(sampling,dim = 1)
        sample_logprob = torch.sum(logprob.permute([1,2,0])*sampling,axis = (1,2))
        rc_loss = torch.mean(torch.mean(mse_loss,axis = (1,2))*sample_logprob)
        rc_loss_raw = torch.mean(mse_loss)
        # print(torch.mean(mse_loss,axis = (1,2)))
        # print(sample_logprob)
        # ### Train with aligner
        # identities,perm = self.aligner.permute_align(raw_seq.cpu().detach().numpy())
        # score = np.mean(identities,axis = 1)
        # best_perm = np.argmax(score)
        # score = torch.from_numpy(identities[best_perm]-self.score_average).to(self.device)
        # self.score_average  = self.score_average * self.decay + (1-self.decay)* np.mean(identities[best_perm])
        # return entropy_loss,rc_loss, torch.mean(-score*sample_logprob)
        return entropy_loss, rc_loss, torch.zeros(1).to(self.device),rc_loss_raw #Hack the alignment loss out
    
    def valid_step(self,batch):
        net = self.encoder
        signal_batch = batch['signal']
        out = net.forward(signal_batch)
        if 'seq' in batch.keys():
            seq = batch['seq']
            seq_len = batch['seq_len'].view(-1)
            errors,perms = net.permute_error(out,
                           seq,
                           seq_len,
                           alphabet = 'N' + self.config.CTC['alphabeta'],
                           beam_size = self.config.CTC['beam_size'],
                           beam_cut_threshold = self.config.CTC['beam_cut_threshold'])
            best_perm = np.argmin(errors)
        else:
            out_seq = torch.argmax(out,dim = 2)
            identities,perms = self.aligner.permute_align(out_seq.cpu().detach().numpy())
            score = np.mean(identities,axis = 1)
            best_perm = np.argmax(score)
        return score[best_perm],perms[best_perm]
    
def main(args):
    class CTC_CONFIG(MM_CONFIG):
        CTC = {"beam_size":5,
               "beam_cut_threshold":0.05,
               "alphabeta": "ACGT"}
    class TRAIN_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "epsilon":0.1,
                 "epsilon_decay":0,
                 "alpha":1.,
                 "beta": 1.,
                 "gamma":0,
                 "preheat":500,
                 "keep_record":5,
                 "decay":args.decay,}
        
    config = TRAIN_CONFIG()
    print("Read chunks and sequence.")
    chunks = np.load(args.chunks,allow_pickle = True)
    print(chunks.shape)
    reference = np.load(args.seq) if args.seq else None
    ref_len = np.load(args.seq_len) if args.seq_len else None
    print("Construct and load the model.")
    model_f = args.model_folder
    dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = 200,shuffle = True, num_workers = 4)
    DEVICE = args.device
    loader = DeviceDataLoader(loader,device = DEVICE)
    if args.retrain:
        config_old = load_config(os.path.join(model_f,"config.toml"))
        config_old.TRAIN = config.TRAIN #Overwrite training config.
        config = config_old
    encoder = CRNN(config)
    decoder = REVCNN(config)
    mm = MM(config)
    aligner = MetricAligner(args.reference)
    t = VAETrainer(loader,encoder,decoder,mm,config,aligner)
    if args.retrain:
        t.load(model_f)
    lr = args.lr
    epoches = args.epoches
    opt_e = torch.optim.Adam(t.encoder_parameters,lr = lr)
    opt_mm = torch.optim.SGD(t.mm.parameters(),lr = lr)
    opt_revcnn = torch.optim.Adam(t.decoder.parameters(),lr = lr)
    COUNT_CYCLE = args.report
    print("Begin training the model.")
    t.train(epoches,[opt_e,opt_mm,opt_revcnn],COUNT_CYCLE,model_f)
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')
    parser.add_argument('-i', '--chunks', required = True,
                        help = "The .npy file contain chunks.")
    parser.add_argument('-o', '--model_folder', required = True,
                        help = "The folder to save folder at.")
    parser.add_argument('-r', '--reference', required = True,
                        help = "The reference fastq file.")
    parser.add_argument('--seq', default = None,
                        help="The .npy file contain the sequence.")
    parser.add_argument('--seq_len', default = None,
                        help="The .npy file contain the sueqnece length.")
    parser.add_argument('--device', default = 'cuda',
                        help="The device used for training, can be cpu or cuda.")
    parser.add_argument('--lr', default = 4e-3, type = float,
                        help="Initial learning rate.")
    parser.add_argument('--batch_size', default = 200, type = int,
                        help="Training batch size.")
    parser.add_argument('--epoches', default = 10, type = int,
                        help = "The number of epoches to train.")
    parser.add_argument('--report', default = 10, type = int,
                        help = "The interval of training rounds to report.")
    parser.add_argument('--load', dest='retrain', action='store_true',
                        help='Load existed model.')
    parser.add_argument('--decay', type = float, default = 0.99,
                        help="The decay factor of the moving average.")
    args = parser.parse_args(sys.argv[1:])
    if not os.path.isdir(args.model_folder):
        os.mkdir(args.model_folder)
    main(args)
