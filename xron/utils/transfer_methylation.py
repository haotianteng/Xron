"""
Created on Tue Aug  3 23:24:39 2021

@author: Haotian Teng
"""
import numpy as np
import os
import sys
import argparse

def main(args):
    meth_f = args.meth
    chunk_f = os.path.join(meth_f,"chunks.npy")
    seq_f = os.path.join(meth_f,"seqs.npy")
    seq_lens_f = os.path.join(meth_f,"seq_lens.npy")
    print("Load data")
    chunk = np.load(chunk_f)
    seq = np.load(seq_f)
    seq_lens = np.load(seq_lens_f)
    print("Transfer sequence into methylation.")
    seq = np.array([x.replace('A','M') for x in seq])
    control_f = args.control
    chunk_control_f = os.path.join(control_f,"chunks.npy")
    seq_control_f = os.path.join(control_f,"seqs.npy")
    seq_lens_control_f = os.path.join(control_f,"seq_lens.npy")
    chunk_c = np.load(chunk_control_f)
    seq_c = np.load(seq_control_f)
    seq_lens_c = np.load(seq_lens_control_f)
    
    print("Control dataset shape:%d, methylation dataset shape:%d"%(chunk.shape[0],chunk_c.shape[0]))
    chunk_all = np.concatenate((chunk_c,chunk),axis = 0)
    seq_all = np.concatenate((seq_c,seq),axis = 0)
    seq_lens_all = np.concatenate((seq_lens_c,seq_lens),axis = 0)
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
                        help = "The methylation folder.")
    parser.add_argument('-c', '--control', required = True,
                        help = "The control folder.")
    parser.add_argument('-o', '--output', required = True,
                        help = "The output folder of the merged dataset.")
    args = parser.parse_args(sys.argv[1:])
    main(args)