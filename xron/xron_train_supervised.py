"""
@author: Haotian Teng
"""
import os 
import sys
import math
import torch
import argparse
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from functools import partial
from xron.xron_model import CRNN, CONFIG
from xron.xron_input import Dataset, ToTensor, NumIndex
from xron.xron_train_base import Trainer, DeviceDataLoader, load_config

class SupervisedTrainer(Trainer):
    def __init__(self,
                 train_dataloader:DataLoader,
                 net:CRNN,
                 config:CONFIG,
                 device:str = None,
                 eval_dataloader:DataLoader = None,
                 logger = None):
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
        self.logger = logger
        
    def train(self,epoches,optimizer,save_cycle,valid_cycle,save_folder,scheduler = None):
        self.save_folder = save_folder
        self._save_config()
        initial_global_step = self.global_step
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                if (i_batch+1)%save_cycle==0:
                    calculate_error = True
                else:
                    calculate_error = False
                loss,error = self.train_step(batch,get_error = calculate_error)
                if torch.isnan(loss):
                    print("NaN loss detected, skip this training step.")
                    continue
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 
                #                                max_norm=self.grad_norm)
                optimizer.step()
                self.losses.append(loss.item())
                self.global_step +=1
                if self.config.TRAIN['run_n_test']:
                    # run n test
                    if self.global_step - initial_global_step > self.config.TRAIN['run_n_test']:
                        return None
                if self.logger is not None:
                    self.logger.add_scalar('Loss/train', loss.item(), self.global_step)
                if (i_batch+1)%save_cycle==0:
                    self.save()
                    with torch.set_grad_enabled(False):
                        eval_i,valid_batch = next(enumerate(self.eval_ds))
                        valid_error,valid_perror = self.valid_step(valid_batch)
                        self.errors.append(valid_error)
                        if self.logger is not None:
                            self.logger.add_scalar('Error/train', error, self.global_step)
                            self.logger.add_scalar('Error/valid', valid_error, self.global_step)
                            self.logger.add_scalar('Error/valid_plain', valid_perror, self.global_step)
                        print("Epoch %d: Batch %d, loss %f, error %f valid_error %f, reducting_error %f"%(epoch_i, i_batch, loss,error, np.mean(valid_error),np.mean(valid_perror)))
                    self.save_loss()
                if (i_batch+1)%valid_cycle==0:
                    with torch.set_grad_enabled(False):
                        eval_i,valid_batch = next(enumerate(self.eval_ds))
                        valid_error,valid_perror = self.valid_step(valid_batch)
                        self.errors.append(valid_error)
                        print("Epoch %d: Batch %d, loss %f, error %f valid_error %f, reducting_error %f"%(epoch_i, i_batch, loss,error, np.mean(valid_error),np.mean(valid_perror)))
                    if scheduler is not None:
                        scheduler.step(valid_error)
                    if self.logger is not None:
                        self.logger.add_scalar('Error/train', error, self.global_step)
                        self.logger.add_scalar('Error/valid', valid_error, self.global_step)
                        self.logger.add_scalar('Error/valid_plain', valid_perror, self.global_step)
                
    def valid_step(self,batch):
        net = self.net
        net.eval()
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
        net.train()
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
        if self.config.TRAIN['label_smooth']:
            label_alternation = self.config.TRAIN['label_alternation'] 
            loss = net.ctc_loss_uncertain_label(out,out_len,seq,seq_len,label_alternation,self.config.TRAIN['label_smooth'])
        error = None
        if get_error:
            error = net.ctc_error(out,
                                  seq,
                                  seq_len,
                                  alphabet = 'N' + self.config.CTC['alphabeta'],
                                  beam_size = self.config.CTC['beam_size'],
                                  beam_cut_threshold = self.config.CTC['beam_cut_threshold'])
        return loss,error

def load_data(chunk_f,seq_f,seqlen_f,mmap_mode = "r",alphabeta = "ACGT",max_n = None):
    chunks = np.load(chunk_f,mmap_mode= mmap_mode)
    reference = np.load(seq_f,mmap_mode= mmap_mode)
    ref_len = np.load(seqlen_f,mmap_mode= mmap_mode)
    if len(chunks) > len(reference):
        print("There are more chunks (%d) than the sequences (%d), it will be cut to equal to the sequences."%(len(chunks),len(reference)))
        chunks = chunks[:len(reference)]
    if max_n is not None:
        chunks = chunks[:max_n]
        reference = reference[:max_n]
        ref_len = ref_len[:max_n]
    nan_filter = ~np.any(np.isnan(chunks),axis=1)
    chunks = chunks[nan_filter]
    reference = reference[nan_filter]
    ref_len = ref_len[nan_filter]
    ref_len = ref_len.astype(np.int64)
    if reference[0].dtype.kind in ['U','S']:
        alphabet_dict = {x:i+1 for i,x in enumerate(alphabeta)}
        dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([NumIndex(alphabet_dict),ToTensor()]))
    else:
        dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    return dataset

def main(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    optimizers = {'Adam':torch.optim.Adam,
                  'AdamW':torch.optim.AdamW,
                  'SGD':torch.optim.SGD,
                  'RMSprop':torch.optim.RMSprop,
                  'Adagrad':torch.optim.Adagrad,
                  'Momentum':partial(torch.optim.SGD,momentum = 0.9)}
    class CTC_CONFIG(CONFIG):
        CTC = {"beam_size":5,
               "beam_cut_threshold":0.05,
               "alphabeta": "ACGTM",
               "mode":"rna"}
    class TRAIN_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "keep_record":5,
                 "eval_size":100,
                 "optimizer":args.optimizer,
                 "embedding_pretrain_model":args.embedding,
                 "run_n_test":args.run_n_test,
                 "label_alternation":[5,1],#5 is the M label and 1 is the A label
                 "label_smooth":args.label_smooth}
    config = TRAIN_CONFIG()
    print("Construct and load the model.")
    model_f = args.model_folder
    if args.retrain:
        config_old = load_config(os.path.join(model_f,"config.toml"))
        config_old.TRAIN = config.TRAIN #Overwrite training config.
        config = config_old
    elif args.config:
        config = load_config(args.config)
    print("Read chunks and sequence.")
    dataset = load_data(args.chunks,args.seq,args.seq_len,mmap_mode = "r",alphabeta = config.CTC['alphabeta'])
    eval_size = config.TRAIN['eval_size']
    if args.eval_chunks:
        print("Read evaluation chunks and sequence.")
        eval_ds = load_data(args.eval_chunks,args.eval_seq,args.eval_seq_len,mmap_mode = "r",alphabeta = config.CTC['alphabeta'],max_n = eval_size*args.batch_size)
    else:
        dataset,eval_ds = torch.utils.data.random_split(dataset,[len(dataset) - eval_size, eval_size], generator=torch.Generator().manual_seed(42))
    loader = data.DataLoader(dataset,batch_size = config.TRAIN['batch_size'],shuffle = True, num_workers = 2)
    loader_eval = data.DataLoader(eval_ds,batch_size = config.TRAIN['batch_size'],shuffle = True, num_workers = 0)
    DEVICE = args.device
    loader = DeviceDataLoader(loader,device = DEVICE)
    loader_eval = DeviceDataLoader(loader_eval,device = DEVICE)
    print("Train dataset: %d batches; Evaluation dataset: %d batches"%(len(loader),len(loader_eval)))
    net = CRNN(config)
    if args.logging:
        writer = SummaryWriter(comment = "LR={LR}_MODEL={MODEL}_ReTrain={ReTrain}_ReOL={ReOL}".format(LR=args.lr,MODEL=os.path.basename(args.model_folder),ReTrain=args.retrain,ReOL=args.retrain_on_last))
        writer.add_text('chunks',args.chunks)
        writer.add_text('seq',args.seq)
    else:
        writer = None
    t = SupervisedTrainer(loader,net,config,eval_dataloader = loader_eval,logger = writer)
    if args.retrain:
        print("Load previous trained model.")
        t.load(model_f)
        if args.retrain_on_last:
            print("Freeze all layers except last %d fully-connected layers"%(args.retrain_on_last))
            assert args.retrain_on_last <= config.FNN['N_Layer'], "Argument retrain_on_last %d is larger than the number of FNN layers %d."%(args.retrain_on_last,config.FNN['N_Layer'])
            n_skip = sum([2 if layer['dropout'] is None else 3 for layer in config.FNN['Layers'][:-args.retrain_on_last] ])
            for layer in t.net.net[:-n_skip]:
                #multiply by 2 here because we have weight and bias.
                for param in layer.parameters():
                    param.requires_grad = False
        if args.reinitialize_methylation:
            print("Reinitialize the methyaltion projection weight.")
            projection = t.net.net[-2]
            stdv = 1./math.sqrt(projection.weight.size(1))
            with torch.no_grad():
                projection.weight[-1] = (torch.rand(projection.weight[-1].shape)-0.5)*stdv
                if projection.bias is not None:
                    projection.bias[-1] = (torch.rand(1)-0.5)*stdv
    elif args.embedding:
        print("Import the embedding model from %s"%(args.embedding))
        t.load(args.embedding)
        for layer in t.net.embedding_layers:
            for param in layer.parameters():
                param.requires_grad = False
    lr = args.lr
    epoches = args.epoches
    optim = optimizers[config.TRAIN['optimizer']](net.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience = 10,factor = 0.3,verbose = True)
    # scheduler = None
    COUNT_CYCLE = args.report
    valid_check_cycle = args.valid_cycle
    print("Begin training the model.")
    t.train(epoches,optim,COUNT_CYCLE,valid_check_cycle,model_f,scheduler = scheduler)
    
    
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
    parser.add_argument('--eval_chunks',default = None,
                        help="The .npy file contain the chunks for evaluation.")
    parser.add_argument('--eval_seq',default = None,
                        help="The .npy file contain the sequence for evaluation.")
    parser.add_argument('--eval_seq_len',default = None,
                        help="The .npy file contain the sequence length for evaluation.")
    parser.add_argument('--embedding', default = None,
                        help="The folder contains the embedding model, if retrain is enable this will have no effect.")
    parser.add_argument('--device', default = 'cuda',
                        help="The device used for training, can be cpu or cuda.")
    parser.add_argument('--lr', default = 4e-3, type = float,
                        help="Initial learning rate.")
    parser.add_argument('--batch_size', default = 200, type = int,
                        help="Training batch size.")
    parser.add_argument('--epoches', default = 10, type = int,
                        help = "The number of epoches to train.")
    parser.add_argument('--report', default = 20, type = int,
                        help = "The interval of training rounds to report.")
    parser.add_argument('--valid_cycle',default = 1000, type = int,
                        help = "The number of cycles to validate the model.")
    parser.add_argument('--load', dest='retrain', action='store_true',
                        help='Load existed model.')
    parser.add_argument('--config', default = None,
                        help = "Training configuration.")
    parser.add_argument('--optimizer', default = "RMSprop",
                        help = "Optimizer to use, can be Adam, AdamW, SGD and RMSprop,\
                            default is RMSprop")
    parser.add_argument('--threads', type = int, default = None,
                        help = "Number of threads used by Pytorch")
    parser.add_argument('--run_n_test', type = int, default = None,
                        help = "Number of batches to run for test if not None.")
    parser.add_argument('--retrain_on_last',type = int, default = None,
                        help = "Number of FNN layers to retrain on, the rest of the layers will be freezed.")
    parser.add_argument('--reinitialize_methylation', dest = "reinitialize_methylation", action = "store_true",
                        help = "Reinitialize the methylation base projection in last FNN layer.")
    parser.add_argument('--nolog', action="store_false", dest="logging",
                        help = "Disable logging.")
    parser.add_argument('--label_smooth', type = float, default = 0.0)
    args = parser.parse_args(sys.argv[1:])
    if args.logging:
        from torch.utils.tensorboard import SummaryWriter
    if args.retrain and args.embedding:
        args.embedding = None
        print("Embedding is being overrided by --load argument.")
    os.makedirs(args.model_folder,exist_ok=True)
    main(args)
