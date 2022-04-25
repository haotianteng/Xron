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
    mm = np.load(mm_f,allow_pickle=True)
    read_ids = mm[-1]
    metas = pd.read_csv(meta_f,delimiter = " ",header = None)
    metas = metas.dropna(how = 'all')
    ids = list(metas[1])
    mask = [True if id in ids else False for id in read_ids]
    mad,med,meth,offsets,scales,read_ids = [list(itertools.compress(x,mask)) for x in mm]
    mms = np.asarray([mad,med])
    chunk_count = [len(list(j)) for i,j in itertools.groupby(ids)]
    mms_full = np.repeat(mms,chunk_count,axis = 1)
    return chunk,seq,seq_lens,mms,mms_full

def main(args):
    chunk_m,seq_m,seq_lens_m= [],[],[]
    print("Read control dataset.")
    if args.control:
        chunk_c,seq_c,seq_lens_c,mms_c,mms_full_c= load_data(args.control)
    else:
        chunk_c,seq_c,seq_lens_c,mms_c,mms_full_c = [],[],[],[],[]
    print("Read methylation dataset.")
    m_size = 0
    if args.meth:
        for i,m in enumerate(args.meth):
            chunk,seq,seq_lens,mms,mms_full = load_data(m)
            print("Transfer sequence into methylation.")
            seq = np.array([x.replace('A','M') for x in seq])
            if args.shift:
                offset = np.mean(mms_c[1,:])-np.mean(mms[1,:])
                offset /= mms_full[0,:]
                chunk += offset[:,None]
            chunk_m.append(chunk)
            seq_m.append(seq)
            seq_lens_m.append(seq_lens)
            m_size += chunk.shape[0]
            print(" Methylation %d dataset shape:%d"%(i+1,chunk.shape[0]))
        print("Control dataset size:%d"%(chunk_c.shape[0]))
        print("Methylation datasets total size:%d"%(m_size))
        chunk_m = np.concatenate(chunk_m,axis = 0)
        seq_m = np.concatenate(seq_m,axis = 0)
        seq_lens_m = np.concatenate(seq_lens_m,axis = 0)
    else:
        chunk_m,seq_m,seq_lens_m = [],[],[]
    if args.cm_ratio:
        size_c = len(chunk_c)
        size_m = len(chunk_m)
        curr_ratio = size_c/float(size_m)
        if curr_ratio > args.cm_ratio:
            shrink_size = int(size_m*args.cm_ratio)
            chunk_c,seq_c,seq_lens_c = chunk_c[:shrink_size],seq_c[:shrink_size],seq_lens_c[:shrink_size]
        elif curr_ratio < args.cm_ratio:
            shrunk_size = int(size_c/args.cm_ratio)
            chunk_m,seq_m,seq_lens_m = chunk_m[:shrink_size],seq_m[:shrink_size],seq_lens_m[:shrink_size]
            
    if args.control and args.meth:
        chunk_all = np.concatenate([chunk_m,chunk_c],axis = 0)
        seq_all = np.concatenate([seq_m,seq_c],axis = 0)
        seq_lens_all = np.concatenate([seq_lens_m,seq_lens_c],axis = 0)
    elif args.control:
        chunk_all,seq_all,seq_lens_all = chunk_c,seq_c,seq_lens_c
    else:
        chunk_all,seq_all,seq_lens_all = chunk_m,seq_m,seq_lens_m
    out_f = args.output
    print("Save the merged dataset.")
    if args.cm_ratio:
        print("Final size: control - %d, methylation - %d, target size ratio %.1f, final size ratio %.1f"%(len(chunk_c),len(chunk_m),args.cm_ratio,len(chunk_c)/float(len(chunk_m))))
    else:
        print("Final size: control - %d, methylation - %d"%(len(chunk_c),len(chunk_m)))
    os.makedirs(out_f,exist_ok = True)
    np.save(os.path.join(out_f,'chunks.npy'),chunk_all)
    np.save(os.path.join(out_f,'seqs.npy'),seq_all)
    np.save(os.path.join(out_f,'seq_lens.npy'),seq_lens_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')
    parser.add_argument('-m', '--meth', default = None,
                        help = "The methylation folders, multiple directories separate by comma.")
    parser.add_argument('-c', '--control', default = None,
                        help = "The control folder.")
    parser.add_argument('-o', '--output', required = True,
                        help = "The output folder of the merged dataset.")
    parser.add_argument('--shift',action = "store_true", dest = "shift",
                        help = "If move the methylation signal according to the shift of the control median value.")
    parser.add_argument('--cm_ratio',default = None, type = float,
                        help = "The size ratio of control/methylation, size will be adjusted to a maximum reads.")
    args = parser.parse_args(sys.argv[1:])
    if not args.control and not args.meth:
        raise ValueError("Neither --control or --meth being specified.")
    if args.cm_ratio and (not args.control or not args.meth):
        raise ValueError("Require both control and methylation dataset being provided when cm_ratio is set.")
    args.meth = args.meth.split(',')
    main(args)
