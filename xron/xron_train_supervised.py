"""
@author: Haotian Teng
"""
import os 
import sys
import torch
import argparse
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader

from xron.xron_model import CRNN, CONFIG
from xron.xron_input import Dataset, ToTensor, NumIndex, rna_filt, dna_filt
from xron.xron_train_base import Trainer, DeviceDataLoader, load_config

class SupervisedTrainer(Trainer):
    def __init__(self,
                 train_dataloader:DataLoader,
                 net:CRNN,
                 config:CONFIG,
                 device:str = None,
                 eval_dataloader:DataLoader = None):
        """

        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader.
        net : CRNN
            A CRNN network instance.
        device: str
            The device used to train the model, can be 'cpu' or 'cuda'.
            Default is None, use cuda device if it's available.
        config: CONFIG
            A CONFIG class contains training configurations. Need to contain
            at least these parameters: keep_record, device and grad_norm.
        eval_dataloader : DataLoader, optional
            Evaluation dataloader, if None training dataloader will be used.
            The default is None.

        Returns
        -------
        None.

        """
        super().__init__(train_dataloader=train_dataloader,
                         nets = {"encoder":net},
                         config = config,
                         device = device,
                         eval_dataloader = eval_dataloader)
        self.net = net
        self.grad_norm = config.TRAIN['grad_norm']
        
    def train(self,epoches,optimizer,save_cycle,save_folder):
        self.save_folder = save_folder
        self._save_config()
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                if (i_batch+1)%save_cycle==0:
                    calculate_error = True
                else:
                    calculate_error = False
                loss,error = self.train_step(batch,get_error = calculate_error)
                if not loss:
                    continue
                optimizer.zero_grad()
                loss.backward()
                if (i_batch+1)%save_cycle==0:
                    self.save()
                    eval_i,valid_batch = next(enumerate(self.eval_ds))
                    valid_error,valid_perror = self.valid_step(valid_batch)
                    print("Epoch %d Batch %d, loss %f, error %f, valid_error %f, reducting_error %f"%(epoch_i, i_batch, loss,np.mean(error),np.mean(valid_error),np.mean(valid_perror)))
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 
                                               max_norm=self.grad_norm)
                optimizer.step()
                self.global_step +=1
                
    def valid_step(self,batch):
        net = self.net
        signal_batch = batch['signal']
        out = net.forward(signal_batch)
        seq = batch['seq']
        seq_len = batch['seq_len'].view(-1)
        error = None
        error = net.ctc_error(out,
                              seq,
                              seq_len,
                              alphabet = 'N' + self.config.CTC['alphabeta'],
                              beam_size = self.config.CTC['beam_size'],
                              beam_cut_threshold = self.config.CTC['beam_cut_threshold'])
        plain_error = net.ctc_error(out,
                                    seq,
                                    seq_len,
                                    alphabet = 'N' + self.config.CTC['alphabeta'],
                                    beam_size = self.config.CTC['beam_size'],
                                    beam_cut_threshold = self.config.CTC['beam_cut_threshold'],
                                    reduction = {'M':'A'})
        return error,plain_error

    def train_step(self,batch,get_error = False):
        net = self.net
        signal_batch = batch['signal']
        if torch.sum(torch.isnan(signal_batch)):
            print("Found NaN input signal.")
            return None,None
        out = net.forward(signal_batch)
        out_len = np.array([out.shape[0]]*out.shape[1],dtype = np.int64)
        out_len = torch.from_numpy(out_len).to(self.device)
        seq = batch['seq']
        seq_len = batch['seq_len'].view(-1)
        loss = net.ctc_loss(out,out_len,seq,seq_len)
        error = None
        if get_error:
            error = net.ctc_error(out,
                                  seq,
                                  seq_len,
                                  alphabet = 'N' + self.config.CTC['alphabeta'],
                                  beam_size = self.config.CTC['beam_size'],
                                  beam_cut_threshold = self.config.CTC['beam_cut_threshold'])
        return loss,error

def main(args):
    class CTC_CONFIG(CONFIG):
        CTC = {"beam_size":1,
               "beam_cut_threshold":0.05,
               "alphabeta": "ACGTM",
               "mode":"rna"}
    class TRAIN_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "keep_record":5}
        
    config = TRAIN_CONFIG()
    print("Read chunks and sequence.")
    chunks = np.load(args.chunks)
    reference = np.load(args.seq)
    ref_len = np.load(args.seq_len)
    print("Construct and load the model.")
    model_f = args.model_folder
    if config.CTC['mode'] == 'rna':
        chunks,reference,ref_len = rna_filt(chunks,reference,ref_len)
    elif config.CTC['mode'] == 'dna':
        chunks,reference,ref_len = dna_filt(chunks,reference,ref_len)
    ref_len = ref_len.astype(np.int64)
    if reference[0].dtype.kind in ['U','S']:
        alphabet_dict = {x:i+1 for i,x in enumerate(TRAIN_CONFIG.CTC['alphabeta'])}
        dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([NumIndex(alphabet_dict),ToTensor()]))
    else:
        dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = 200,shuffle = True, num_workers = 4)
    DEVICE = args.device
    loader = DeviceDataLoader(loader,device = DEVICE)
    if args.retrain:
        config_old = load_config(os.path.join(model_f,"config.toml"))
        config_old.TRAIN = config.TRAIN #Overwrite training config.
        config = config_old
    elif args.config:
        config = load_config(args.config)
    net = CRNN(config)
    t = SupervisedTrainer(loader,net,config)
    if args.retrain:
        t.load(model_f)
    lr = args.lr
    epoches = args.epoches
    optimizer = torch.optim.Adam(net.parameters(),lr = lr)
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
    parser.add_argument('--seq', required = True,
                        help="The .npy file contain the sequence.")
    parser.add_argument('--seq_len', required = True,
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
    parser.add_argument('--config', default = None,
                        help = "Training configuration.")
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.model_folder,exist_ok=True)
    main(args)
