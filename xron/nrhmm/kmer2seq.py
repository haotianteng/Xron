"""
Created on Fri Jun 17 23:00:01 2022

@author: Haotian Teng
"""
import os 
import sys
import toml
import argparse
import numpy as np
import itertools
from tqdm import tqdm
def kmers2seq(kmers,idx2kmer):
    merged = [g for g,_ in itertools.groupby(kmers)]
    seqs = [idx2kmer[x][0] for x in merged]
    return ''.join(seqs) + idx2kmer[merged[-1]][1:]

def transfer(args):
    kmers = np.load(os.path.join(args.input,"kmers.npy"),mmap_mode = "r")
    durations = np.load(os.path.join(args.input,"durations.npy"),mmap_mode = "r")
    config = toml.load(os.path.join(args.input,"config.toml"))
    seqs,seq_lens = [],[]
    seqs = [kmers2seq(x[:durations[i]],config['idx2kmer']) for i,x in tqdm(enumerate(kmers))]
    seq_lens = [len(x) for x in seqs]
    np.save(os.path.join(args.input,'seqs.npy'),seqs)
    np.save(os.path.join(args.input,'seq_lens.npy'),seq_lens)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer the kmer-seq to sequences.')
    parser.add_argument('-i', '--input', required = True,
                        help = "The folder contains the kmers.npy and durations.npy files.")
    args = parser.parse_args(sys.argv[1:])
    transfer(args)