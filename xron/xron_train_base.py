"""
Created on Thu Mar 11 16:07:12 2021

@author: Haotian Teng
"""
import os
import toml
import torch
import numpy as np
from typing import Union,Dict
from torch.utils.data.dataloader import DataLoader
from xron.xron_model import CRNN, REVCNN, CONFIG,DECODER_CONFIG,MM


class Trainer(object):
    def __init__(self,
                 train_dataloader:DataLoader,
                 nets:Dict[str,Union[CRNN,REVCNN,MM]],
                 config:Union[CONFIG,DECODER_CONFIG],
                 device:str = None,
                 eval_dataloader:DataLoader = None):
        """

        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader.
        nets : Dict[str,Union[CRNN,REVCNN]]
            A CRNN or REVCNN network instance.
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
        self.train_ds = train_dataloader
        self.device = self._get_device(device)
        if eval_dataloader is None:
            self.eval_ds = self.train_ds
        else:
            self.eval_ds = eval_dataloader
        self.nets = nets
        for net in self.nets.values():
            net.to(self.device)
        self.global_step = 0
        self.save_list = []
        self.keep_record = config.TRAIN['keep_record']
        self.grad_norm = config.TRAIN['grad_norm']
        self.config = config
        self.losses = []
        self.errors = []
    
    def _get_device(self,device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _update_records(self):
        record_file = os.path.join(self.save_folder,'records.toml')
        with open(record_file,'w+') as f:
            toml.dump(self.records,f)

    def save(self):
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        if os.path.isfile(ckpt_file):
            with open(ckpt_file,'r') as f:
                f.read()
                for line in f:
                    ckpt = line.strip().split(':')
                    if ckpt[0].startswith("checkpoint"):
                        self.save_list.append(ckpt)
        current_ckpt = 'ckpt-'+str(self.global_step)
        model_file = os.path.join(self.save_folder,current_ckpt)
        self.save_list.append(current_ckpt)
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        if len(self.save_list) > self.keep_record:
            os.remove(os.path.join(self.save_folder,self.save_list[0]))
            self.save_list = self.save_list[1:]
        if os.path.isfile(model_file):
            os.remove(model_file)
        with open(ckpt_file,'w+') as f:
            f.write("latest checkpoint:" + current_ckpt + '\n')
            for path in self.save_list:
                f.write("checkpoint file:" + path + '\n')
        net_dict = {key:net.state_dict() for key,net in self.nets.items()}
        torch.save(net_dict,model_file)
    
    def save_loss(self):
        loss_file = os.path.join(self.save_folder,'losses.csv')
        error_file = os.path.join(self.save_folder,'errors.csv')
        if len(self.losses):
            with open(loss_file,'a+') as f:
                f.write('\n'.join([str(x) for x in self.losses]))
                f.write('\n')
        if len(self.errors):
            with open(error_file,'a+') as f:
                f.write('\n'.join([str(x) for x in self.errors]))
                f.write('\n')
        self.losses = []
        self.errors = []
    
    def _save_config(self):
        config_file = os.path.join(self.save_folder,'config.toml')
        config_modules = [x for x in self.config.__dir__() if not x .startswith('_')][::-1]
        config_dict = {x:getattr(self.config,x) for x in config_modules}
        with open(config_file,'w+') as f:
            toml.dump(config_dict,f)
    
    def load(self,save_folder):
        self.save_folder = save_folder
        ckpt_file = os.path.join(save_folder,'checkpoint')
        with open(ckpt_file,'r') as f:
            latest_ckpt = f.readline().strip().split(':')[1]
            self.global_step = int(latest_ckpt.split('-')[1])
        ckpt = torch.load(os.path.join(save_folder,latest_ckpt),
                          map_location=self.device)
        for key,net in self.nets.items():
            net.load_state_dict(ckpt[key])
            net.to(self.device)

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

def load_config(config_file):
    class CONFIG(object):
        pass
    with open(config_file,'r') as f:
        config_dict = toml.load(f)
    config = CONFIG()
    for k,v in config_dict.items():
        setattr(config,k,v)
    return config