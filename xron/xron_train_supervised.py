"""
Created on Thu Mar  4 19:27:19 2021

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
from xron.xron_input import Dataset, ToTensor

class Trainer(object):
    def __init__(self,
                 train_dataloader:DataLoader,
                 net:CRNN,
                 keep_record:int = 5,
                 eval_dataloader:DataLoader = None,
                 device:str = None):
        """

        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader.
        net : CRNN
            A CRNN network instance.
        keep_record : int, optional
            The latest n checkpoints to save for each training routine.. The default is 5.
        eval_dataloader : DataLoader, optional
            Evaluation dataloader, if None training dataloader will be used.
            The default is None.
        device : str, optional
            The name of the device, can be one of the following:"cpu", "cuda". 
            If None, cuda will be used if available. The default is None.

        Returns
        -------
        None.

        """
        self.train_ds = train_dataloader
        self.device = self._get_device(device)
        if eval_dataloader is None:
            self.eval_ds = self.train_ds
        else:
            self.eval_ds = eval_dataloader
        self.net = net
        self.net.to(self.device)
        self.global_step = 0
        self.keep_record = keep_record
        self.save_list = []
    
    def _get_device(self,device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
        
    def save(self):
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        current_ckpt = 'ckpt-'+str(self.global_step)
        model_file = os.path.join(self.save_folder,current_ckpt)
        self.save_list.append(current_ckpt)
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        if len(self.save_list) > self.keep_record:
            os.remove(os.path.join(self.save_folder,self.save_list[0]))
            self.save_list = self.save_list[1:]
        with open(ckpt_file,'w+') as f:
            f.write("latest checkpoint:" + current_ckpt + '\n')
            for path in self.save_list:
                f.write("checkpoint file:" + path + '\n')
        torch.save(self.net.state_dict(),model_file)
    
    def load(self,save_folder):
        self.save_folder = save_folder
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        with open(ckpt_file,'r') as f:
            latest_ckpt = f.readline().strip().split(':')[1]
            self.global_step = int(latest_ckpt.split('-')[1])
        self.net.load_state_dict(torch.load(os.path.join(save_folder,latest_ckpt),map_location=self.device))
        self.net.to(self.device)
        
    def train(self,epoches,optimizer,save_cycle,save_folder):
        self.save_folder = save_folder
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                if i_batch+1%save_cycle==0:
                    calculate_error = True
                else:
                    calculate_error = False
                loss,error = self.train_step(batch,get_error = calculate_error)
                optimizer.zero_grad()
                loss.backward()
                if i_batch+1%save_cycle==0:
                    self.save()
                    eval_i,valid_batch = next(enumerate(self.eval_ds))
                    valid_error = self.valid_step(valid_batch)
                    print(valid_error)
                    print("Epoch %d Batch %d, loss %f, error %f, valid_error %f"%(epoch_i, i_batch, loss,np.mean(error),np.mean(valid_error)))
                optimizer.step()
                self.global_step +=1
                
    def valid_step(self,batch):
        net = self.net
        signal_batch = batch['signal']
        out = net.forward(signal_batch)
        seq = batch['seq']
        seq_len = batch['seq_len'].view(-1)
        error = None
        error = net.ctc_error(out,seq,seq_len)
        return error

    def train_step(self,batch,get_error = False):
        net = self.net
        signal_batch = batch['signal']
        batch_size = signal_batch.shape[0]
        out = net.forward(signal_batch)
        out_len = np.array([out.shape[1]]*batch_size,dtype = np.int16)
        out_len = torch.from_numpy(out_len).to(self.device)
        seq = batch['seq']
        seq_len = batch['seq_len'].view(-1)
        loss = net.ctc_loss(out,out_len,seq,seq_len)
        error = None
        if get_error:
            error = net.ctc_error(out,seq,seq_len)
        return loss,error

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device = None):
        self.dataloader = dataloader
        if device is None:
            device = self.get_default_device()
        else:
            device = torch.device(device)
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield self._to_device(b, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)
    
    def _to_device(self,data,device):
        if isinstance(data, (list,tuple)):
            return [self._to_device(x,device) for x in data]
        if isinstance(data, (dict)):
            temp_dict = {}
            for key in data.keys():
                temp_dict[key] = self._to_device(data[key],device)
            return temp_dict
        return data.to(device, non_blocking=True)
    
    def get_default_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def main(args):
    print("Read chunks and sequence.")
    chunks = np.load(args.chunks)
    reference = np.load(args.seq)
    ref_len = np.load(args.seq_len)
    print("Construct and load the model.")
    model_f = args.model_folder
    dataset = Dataset(chunks,seq = reference,seq_len = ref_len,transform = transforms.Compose([ToTensor()]))
    loader = data.DataLoader(dataset,batch_size = 200,shuffle = True, num_workers = 4)
    DEVICE = args.device
    loader = DeviceDataLoader(loader)
    config = CONFIG()
    net = CRNN(config)
    t = Trainer(loader,net,device = DEVICE)
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
    parser.add_argument('--epoches', default = 10, type = int,
                        help = "The number of epoches to train.")
    parser.add_argument('--report', default = 10, type = int,
                        help = "The interval of training rounds to report.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
