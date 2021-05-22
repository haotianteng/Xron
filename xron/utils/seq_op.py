"""
Created on Mon Apr 19 13:59:01 2021

@author: Haotian Teng
"""
from numpy import ndarray
import numpy as np
from typing import List

def raw2seq(raw:ndarray, blank_symbol:int = 0)->List[ndarray]:
    """
    Transfer raw ctc sequence by removing the consecutive repeat and blank symbol.

    Parameters
    ----------
    raw : ndarray
        The [N,C,L] raw ctc sequence, where N is the batch size and L is the 
        sequence length.
    blank_symbol : int, optional
        The index of the blank symbol. The default is 0.

    Returns
    -------
    List[ndarray]
        Length N list contains Transfered sequences.

    """
    mask = np.diff(raw).astype(bool)
    mask = np.insert(mask,0,True,axis = 1)
    np.ma.array(raw,mask = mask).tolist()
    seqs = np.ma.array(raw,mask = np.logical_not(mask)).tolist(None)
    return [[x for x in seq if x is not None and x!=blank_symbol] for seq in seqs]