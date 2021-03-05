"""
Created on Thu Feb 25 09:37:04 2021
The generative model of Xron
@author: Haotian Teng
"""
from torch import nn
class Wavenet(nn.Module):
    def __init__(self, 
                 k = 5, 
                 n_base = 5):
        