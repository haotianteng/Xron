"""
This script train a VQ-VAE style embedding network on Nanopore sequencing signal.
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
from xron.xron_input import Dataset, ToTensor, NumIndex,rna_filt,dna_filt
from xron.xron_train_base import Trainer, DeviceDataLoader, load_config
from xron.xron_model import REVCNN,DECODER_CONFIG,CRNN,CRITIC_CONFIG,CRITIC,MM_CONFIG,MM
from xron.utils.vq import vq
from xron.xron_label import MetricAligner
from torch.distributions.one_hot_categorical import OneHotCategorical as OHC

class VQVAETrainer(Trainer):
    def __init__(self,
                 train_dataloader:DataLoader,
                 encoder:CRNN,
                 decoder:REVCNN,
                 mm:MM,
                 config:Union[DECODER_CONFIG,MM_CONFIG],
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
        self.train_config = config.TRAIN
        self.global_step = 0
        self.score_average = 0
        self.nn_embd = vq
        self.mse_loss = torch.nn.MSELoss(reduction = "mean")
        self.records = {'rc_losses':[],
                        'rc_valid':[],
                        'embedding_loss':[],
                        'commitment_loss':[]}
    @property
    def encoder(self):
        return self.nets["encoder"]
    
    @property
    def decoder(self):
        return self.nets["decoder"]

    @property
    def mm(self):
        return self.nets["mm"]
    
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
            A list of three optimizers, the first one is optimizer training the
            encoder parameters, the second one for decoder parameters and the
            third one is the optimizer for the embedding.
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
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                losses = self.train_step(batch)
                loss = losses[0] + losses[1] + self.train_config['alpha']*losses[2]
                for opt in optimizers:
                    opt.zero_grad()
                loss.backward()
                for opt in optimizers:
                    opt.step()
                if (self.global_step+1)%save_cycle==0:
                    self.save()
                    eval_i,valid_batch = next(enumerate(self.eval_ds))
                    with torch.no_grad():
                        valid_rc = self.valid_step(valid_batch)
                    records["rc_valid"].append(valid_rc.detach().cpu().numpy()[()])
                    records['rc_losses'].append(losses[0].detach().cpu().numpy()[()])
                    records['embedding_loss'].append(losses[1].detach().cpu().numpy()[()])
                    records['commitment_loss'].append(losses[2].detach().cpu().numpy()[()])
                    print("Epoch %d Batch %d, rc_loss %f, embedding_loss %f, validation rc %f"%(epoch_i, i_batch, losses[0], losses[1],valid_rc))
                    self._update_records()
                losses = None
                torch.nn.utils.clip_grad_norm_(self.parameters, 
                                               max_norm=self.grad_norm)
                self.global_step +=1
        
    def train_step(self,batch):
        encoder = self.encoder
        decoder = self.decoder
        embedding = self.mm.level_embedding
        signal = batch['signal']
        q = encoder.forward_wo_fnn(signal) #[N,C,L]
        e,e_shadow = self.nn_embd(q.permute([0,2,1]),embedding.weight) #[N,L,C] -> [N,C,L]
        e = e.permute([0,2,1])
        e_shadow = e_shadow.permute([0,2,1])
        sg_q = q.detach()
        sg_e = e.detach()
        # q = q + torch.normal(torch.zeros(q.shape),std = self.train_config['sigma'])
        rc_signal = decoder.forward(e).permute([0,2,1]) #[N,L,C] -> [N,C,L]
        rc_loss = self.mse_loss(rc_signal,signal)
        embedding_loss = self.mse_loss(sg_q,e_shadow)
        commitment_loss = self.mse_loss(sg_e,q)
        return rc_loss, embedding_loss, commitment_loss
    
    def valid_step(self,batch):
        rc_loss,_,_ = self.train_step(batch)
        return rc_loss
    
def main(args):
    class CTC_CONFIG(MM_CONFIG):
        CTC = {"beam_size":5,
               "beam_cut_threshold":0.05,
               "alphabeta": "ACGTM",
               "mode":"rna"}
    class TRAIN_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "epsilon":0.1,
                 "epsilon_decay":0,
                 "alpha":1.0, #Entropy loss scale factor
                 "keep_record":5,
                 "decay":args.decay,
                 "diff_signal":args.diff}
        
    config = TRAIN_CONFIG()
    config.PORE_MODEL["N_BASE"] = len(config.CTC["alphabeta"])
    print("Read chunks and sequence.")
    chunks = np.load(args.chunks,allow_pickle = True)
    reference = np.load(args.seq) if args.seq else None
    ref_len = np.load(args.seq_len) if args.seq_len else None
    print("Construct and load the model.")
    model_f = args.model_folder
    if reference is not None and (reference[0].dtype.kind in ['U','S']):
        if config.CTC['mode'] == 'rna':
            chunks,reference,ref_len = rna_filt(chunks,reference,ref_len)
        elif config.CTC['mode'] == 'dna':
            chunks,reference,ref_len = dna_filt(chunks,reference,ref_len)
        alphabet_dict = {x:i+1 for i,x in enumerate(TRAIN_CONFIG.CTC['alphabeta'])}
        dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([NumIndex(alphabet_dict),ToTensor()]))
    else:
        dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True, num_workers = 4)
    DEVICE = args.device
    loader = DeviceDataLoader(loader,device = DEVICE)
    if args.retrain:
        config_old = load_config(os.path.join(model_f,"config.toml"))
        config_old.TRAIN = config.TRAIN #Overwrite training config.
        config = config_old
    encoder = CRNN(config)
    decoder = REVCNN(config)
    mm = MM(config)
    t = VQVAETrainer(loader,encoder,decoder,mm,config)
    if args.retrain:
        t.load(model_f)
    lr = args.lr
    epoches = args.epoches
    opt = torch.optim.Adam(chain(t.encoder.parameters(),t.decoder.parameters(),t.mm.level_embedding.parameters()),lr = lr)
    COUNT_CYCLE = args.report
    print("Begin training the model.")
    t.train(epoches,[opt],COUNT_CYCLE,model_f)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with chunks and sequence file')
    parser.add_argument('-i', '--chunks', required = True,
                        help = "The .npy file contain chunks.")
    parser.add_argument('-o', '--model_folder', required = True,
                        help = "The folder to save folder at.")
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
    parser.add_argument('--diffSig',action="store_true",dest = "diff",
                        help="If the input chunks are diffrential signal.")
    args = parser.parse_args(sys.argv[1:])
    if not os.path.isdir(args.model_folder):
        os.mkdir(args.model_folder)
    main(args)
