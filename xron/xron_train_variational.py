"""
@author: Haotian Teng
"""
import os 
import sys
import torch
import argparse
import numpy as np
from itertools import chain
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from xron.xron_input import Dataset, ToTensor
from xron.xron_train_base import Trainer, DeviceDataLoader, load_config
from xron.xron_model import REVCNN,DECODER_CONFIG,CRNN
from xron.xron_label import MetricAligner
from torch.distributions.one_hot_categorical import OneHotCategorical as OHC

class VAETrainer(Trainer):
    def __init__(self,
                 train_dataloader:DataLoader,
                 encoder:CRNN,
                 decoder:REVCNN,
                 config:DECODER_CONFIG,
                 aligner:MetricAligner,
                 device:str = None,
                 eval_dataloader:DataLoader = None):
        """

        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader.
        net : REVCNN
            A REVCNN network instance.
        device: str
            The device used to train the model, can be 'cpu' or 'cuda'.
            Default is None, use cuda device if it's available.
        config: DECODER_CONFIG
            A CONFIG class contains unsupervised training configurations. Need 
            to contain at least these parameters: keep_record, device and 
            grad_norm.
        eval_dataloader : DataLoader, optional
            Evaluation dataloader, if None training dataloader will be used.
            The default is None.

        """
        super().__init__(train_dataloader=train_dataloader,
                         nets = {"encoder":encoder,
                                 "decoder":decoder},
                         config = config,
                         device = device,
                         eval_dataloader = eval_dataloader)
        self.encoder = encoder
        self.decoder = decoder
        self.aligner = aligner
        params = [encoder.parameters(),decoder.parameters()]
        self.parameters = chain(*params)
        self.decay = config.TRAIN["decay"]
        self.global_step = 0
        self.awake = False
        self.score_average = 0
        self.wake_sleep_cycle = config.TRAIN['Sleep_Wake_Cycle']
        self.records = {'rc_losses':[],
                        'rc_losses_encoder':[],
                        'entropy_losses':[],
                        'alignment_score':[],
                        'valid_alignment':[]}
    def train(self,epoches,optimizer,save_cycle,save_folder):
        self.save_folder = save_folder
        self._save_config()
        records = self.records
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                if (self.global_step+1)%self.wake_sleep_cycle == 0:
                    self.awake = not self.awake
                    for param in self.encoder.parameters():
                        param.requires_grad = self.awake
                    for param in self.decoder.parameters():
                        param.requires_grad = not self.awake
                losses = self.train_step(batch,phase = self.awake)
                loss = sum(losses)
                optimizer.zero_grad()
                loss.backward()
                if (self.global_step+1)%save_cycle==0:
                    self.save()
                    eval_i,valid_batch = next(enumerate(self.eval_ds))
                    valid_error,valid_perm = self.valid_step(valid_batch)
                    records['entropy_losses'].append(losses[0].detach().cpu().numpy()[()])
                    records['valid_alignment'].append(valid_error)
                    print(losses)
                    print(valid_error)
                    if self.awake:
                        print("Epoch %d Batch %d, entropy_loss %f, rc_loss %f, valid_error %f, perm:%s, alignment_loss %f"%(epoch_i, i_batch, losses[0], losses[1], valid_error, valid_perm, losses[2]))
                        records['alignment_score'].append(losses[2].detach().cpu().numpy()[()])
                        records['rc_losses_encoder'].append(losses[1].detach().cpu().numpy()[()])
                    else:
                        print("Epoch %d Batch %d, entropy_loss %f, rc_loss %f, valid_error %f, perm:%s"%(epoch_i, i_batch, losses[0], losses[1], valid_error, valid_perm))
                        records['rc_losses'].append(losses[1].detach().cpu().numpy()[()])
                    self._update_records()
                torch.nn.utils.clip_grad_norm_(self.parameters, 
                                               max_norm=self.grad_norm)
                optimizer.step()
                self.global_step +=1
        
    def train_step(self,batch,phase = 0):
        encoder = self.encoder
        decoder = self.decoder
        signal = batch['signal']
        logprob = encoder.forward(signal) #[L,N,C]
        m = OHC(logits = logprob)
        sampling = m.sample().permute([1,2,0]) #[L,N,C]->[N,C,L]
        rc_signal = decoder.forward(sampling).permute([0,2,1]) #[N,L,C] -> [N,C,L]
        mse_loss = decoder.mse_loss(rc_signal,signal)
        entropy_loss = decoder.entropy_loss(logprob.permute([1,2,0]),sampling.max(dim = 1)[1])
        if phase == 0: #Phase 0 when we update the decoder
            rc_loss = torch.mean(mse_loss)
            return entropy_loss,rc_loss
        else: #Phase 1 when we update the encoder.
            raw_seq = torch.argmax(sampling,dim = 1)
            sample_logprob = torch.sum(logprob.permute([1,2,0])*sampling,axis = (1,2))
            rc_loss = torch.mean(torch.mean(mse_loss,axis = (1,2))*sample_logprob)
            
            # ### Train with aligner
            # identities,perm = self.aligner.permute_align(raw_seq.cpu().detach().numpy())
            # score = np.mean(identities,axis = 1)
            # best_perm = np.argmax(score)
            # score = torch.from_numpy(identities[best_perm]-self.score_average).to(self.device)
            # self.score_average  = self.score_average * self.decay + (1-self.decay)* np.mean(identities[best_perm])
            # return entropy_loss,rc_loss, torch.mean(-score*sample_logprob)
            
            return entropy_loss, rc_loss,0

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
    class CTC_CONFIG(DECODER_CONFIG):
        CTC = {"beam_size":5,
               "beam_cut_threshold":0.05,
               "alphabeta": "ACGT"}
    class TRAIN_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "keep_record":5,
                 "decay":args.decay,
                 "Sleep_Wake_Cycle":args.sleep_cycle}
        
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
    aligner = MetricAligner(args.reference)
    t = VAETrainer(loader,encoder,decoder,config,aligner)
    if args.retrain:
        t.load(model_f)
    lr = args.lr
    epoches = args.epoches
    optimizer = torch.optim.Adam(t.parameters,lr = lr)
    COUNT_CYCLE = args.report
    print("Begin training the model.")
    t.train(epoches,optimizer,COUNT_CYCLE,model_f)
    
        
        
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
    parser.add_argument('--sleep_cycle', default = 50, type = int,
                        help = "The sleep wake cycle.")
    parser.add_argument('--load', dest='retrain', action='store_true',
                        help='Load existed model.')
    parser.add_argument('--decay', type = float, default = 0.99,
                        help="The decay factor of the moving average.")
    args = parser.parse_args(sys.argv[1:])
    if not os.path.isdir(args.model_folder):
        os.mkdir(args.model_folder)
    main(args)
