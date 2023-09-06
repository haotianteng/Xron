"""
Created on Fri Nov 19 22:51:28 2021

@author: Haotian Teng
"""
import numpy as np
from itertools import product,repeat

dataset = "/home/heavens/bridge_scratch/m6A_Nanopore/merged_dataset_diff/"
chunks_f = dataset + "chunks.npy"
seq_lens_f = dataset + "seq_lens.npy"
seqs_f = dataset + "seqs.npy"
chunks = np.load(chunks_f)
seq_lens = np.load(seq_lens_f)
seqs = np.load(seqs_f)

def check_ratio(sequences):
    control_count = 0
    m_count = 0
    other_count = 0
    for seq in sequences:
        if 'A' in seq:
            if 'M' in seq:
                raise("ValueError: The read has both M and A in sequence.")
            control_count +=1
        elif 'M' in seq:
            m_count += 1
        else:
            other_count +=1
    return control_count,m_count, control_count + m_count+other_count

def check_Mkmer(sequences,k = 5):
    N = ['M','C','G','T']
    kmers = product(*repeat(N,k))
    kmer_dict = {}
    for kmer in kmers:
        kmer_dict[''.join(kmer)] = 0
    for seq in sequences:
        occurs = [i for i,c in enumerate(seq) if c=='M']
        for o in occurs:
            curr_kmer = seq[o:o+k]
            if len(curr_kmer) == k:
                kmer_dict[curr_kmer] +=1
    return kmer_dict

c,m,all = check_ratio(seqs)
kmer_dict = check_Mkmer(seqs)