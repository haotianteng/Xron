"""
Created on Tue Aug  3 23:24:39 2021

@author: Haotian Teng
"""
import numpy as np
import os
import sys
import pandas as pd
import argparse
import itertools
from typing import Tuple,Iterable
def load_data(prefix:str)->Tuple[np.array,Iterable,np.array,np.array,np.array]:
    """
    Load data from the prefix folder

    Parameters
    ----------
    prefix : str
        The folder contains the chunk data and meta information.

    Returns
    -------
    chunk : np.array[N,L]
        Contains the N chunks which length L.
    seq : Iterable[N]
        Contains N sequences corresponding.
    seq_lens : TYPE
        DESCRIPTION.
    mms : TYPE
        DESCRIPTION.
    mms_full : TYPE
        DESCRIPTION.

    """
    chunk_f = os.path.join(prefix,"chunks.npy")
    seq_f = os.path.join(prefix,"seqs.npy")
    seq_lens_f = os.path.join(prefix,"seq_lens.npy")
    mm_f = os.path.join(prefix,"mm.npy")
    meta_f = os.path.join(prefix,"meta.csv")
    chunk = np.load(chunk_f)
    seq = np.load(seq_f)
    seq_lens = np.load(seq_lens_f)
    mad,med,meth = np.load(mm_f,allow_pickle=True)
    mms = np.asarray([mad,med])
    metas = pd.read_csv(meta_f,delimiter = " ",header = None)
    metas = metas.dropna(how = 'all')
    ids = list(metas[1])
    chunk_count = [len(list(j)) for i,j in itertools.groupby(ids)]
    mms_full = np.repeat(mms,chunk_count,axis = 1)
    return chunk,seq,seq_lens,mms,mms_full

def main(args):
    chunk_all,seq_all,seq_len_all = [],[],[]
    print("Read control dataset.")
    chunk_c,seq_c,seq_lens_c,mms_c,mms_full_c= load_data(args.control)
    chunk_all.append(chunk_c)
    seq_all.append(seq_c)
    seq_len_all.append(seq_lens_c)
    print("Read methylation dataset.")
    for i,m in enumerate(args.meth):
        chunk,seq,seq_lens,mms,mms_full = load_data(m)
        print("Transfer sequence into methylation.")
        seq = np.array([x.replace('A','M') for x in seq])
        if args.shift:
            offset = np.mean(mms_c[1,:])-np.mean(mms[1,:])
            offset /= mms_full[0,:]
            chunk += offset[:,None]
        chunk_all.append(chunk)
        seq_all.append(seq)
        seq_len_all.append(seq_lens)
        print(" Methylation %d dataset shape:%d"%(i+1,chunk.shape[0]))
    print("Control dataset shape:%d"%(chunk_c.shape[0]))
    chunk_all = np.concatenate(chunk_all,axis = 0)
    seq_all = np.concatenate(seq_all,axis = 0)
    seq_lens_all = np.concatenate(seq_len_all,axis = 0)
    out_f = args.output
    print("Save the merged dataset.")
    os.makedirs(out_f,exist_ok = True)
    np.save(os.path.join(out_f,'chunks.npy'),chunk_all)
    np.save(os.path.join(out_f,'seqs.npy'),seq_all)
    np.save(os.path.join(out_f,'seq_lens.npy'),seq_lens_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')
    parser.add_argument('-m', '--meth', required = True,
                        help = "The methylation folders, multiple directories separate by comma.")
    parser.add_argument('-c', '--control', required = True,
                        help = "The control folder.")
    parser.add_argument('-o', '--output', required = True,
                        help = "The output folder of the merged dataset.")
    parser.add_argument('--shift',action = "store_true", dest = "shift",
                        help = "If move the methylation signal according to the shift of the control median value.")
    args = parser.parse_args(sys.argv[1:])
    args.meth = args.meth.split(',')
    main(args)