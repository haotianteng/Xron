"""
@author: Haotian Teng
"""
import torch
from torch import nn
from functools import partial
from typing import List


Conv1dk1 = partial(nn.Conv1d,kernel_size = 1)
RevConv1dk1 = partial(nn.ConvTranspose1d, kernel_size = 1)

class Res1d(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 stride: int = 1,
                 activation: nn.Module = nn.SiLU,
                 batch_norm: nn.Module = nn.BatchNorm1d):
        super().__init__()
        self.self_map = nn.Conv1d(in_channels,
                                  out_channels,
                                  stride = stride,
                                  kernel_size = stride)
        
        self.conv1 = Conv1dk1(in_channels, out_channels,stride=1)
        self.bn1 = batch_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, 
                               out_channels, 
                               kernel_size = kernel_size,
                               stride = stride,
                               padding = (kernel_size-stride)//2)
        self.bn2 = batch_norm(out_channels)
        self.activation = activation(inplace = True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.self_map(x)
        out = self.conv1(x)
        if isinstance(self.bn1,nn.LayerNorm):
            out = self.bn1(out.permute(0,2,1)).permute(0,2,1)
        else:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if isinstance(self.bn2,nn.LayerNorm):
            out = self.bn2(out.permute(0,2,1)).permute(0,2,1)
        else:
            out = self.bn2(out)
        out = self.activation(out)
        out += x0
        return out

class RevRes1d(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 stride: int = 1,
                 activation: nn.Module = nn.SiLU,
                 batch_norm: nn.Module = nn.BatchNorm1d):
        super().__init__()
        self.self_map = nn.ConvTranspose1d(in_channels,
                                           out_channels,
                                           kernel_size = stride,
                                           stride = stride)
        self.conv1 = Conv1dk1(in_channels, out_channels,stride=1)
        self.bn1 = batch_norm(out_channels)
        self.conv2 = nn.ConvTranspose1d(out_channels, 
                                        out_channels, 
                                        kernel_size = kernel_size,
                                        stride = stride,
                                        padding = (kernel_size-stride)//2)
        self.bn2 = batch_norm(out_channels)
        self.activation = activation(inplace = True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.self_map(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        out += x0
        return out
    
class BidirectionalRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 cell: nn.Module = nn.LSTM):
        super(BidirectionalRNN,self).__init__()
        self.rnn = cell(input_size,hidden_size,num_layers,bidirectional = True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output,(hn,cn) = self.rnn(x)
        return output

class Permute(nn.Module):
    def __init__(self,perm:List):
        super().__init__()
        self.perm = perm
    def forward(self,x:torch.Tensor):
        return x.permute(self.perm)