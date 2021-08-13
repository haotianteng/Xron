"""
Created on Fri Aug 13 16:03:48 2021

@author: Haotian Teng
"""
import os
import numpy as np
from matplotlib import pyplot as plt

def map_base(string):
    base_dict = {'A':1,'C':2,'G':3,'T':4}
    return np.asarray([base_dict[b] for b in string])


bonito_ref = "/home/heavens/twilight_data1/S10_DNA/20170322_c4_watermanag_S10/bonito_output/ctc_data/references.npy"
bonito_len = "/home/heavens/twilight_data1/S10_DNA/20170322_c4_watermanag_S10/bonito_output/ctc_data/reference_lengths.npy"
ref = np.load(bonito_ref)
ref_len = np.load(bonito_len)
plt.hist(ref_len)

chunk_f = "/home/heavens/twilight_hdd1/m6A_Nanopore/control.zymo/guppy_result/extracted/chunks.npy"
seq_f = "/home/heavens/twilight_hdd1/m6A_Nanopore/control.zymo/guppy_result/extracted/seqs.npy"
seq_len_f = "/home/heavens/twilight_hdd1/m6A_Nanopore/control.zymo/guppy_result/extracted/seq_lens.npy"
seq = np.load(seq_f)
seq_len = np.load(seq_len_f)
chunk = np.load(chunk_f)
plt.figure()
plt.hist(seq_len,bins = 80)
mask = np.logical_and(seq_len<300,seq_len>100)
seq = seq[mask]
seq_len_filt = seq_len[mask]
plt.figure()
plt.hist(seq_len,bins = 80)
pad_w = np.max(seq_len)
seq_filt = [map_base(s) for s in seq]
seq_filt = [np.pad(x,(0,pad_w-len(x))) for x in seq_filt]
seq_filt = np.asarray(seq_filt)
os.makedirs("/home/heavens/twilight_hdd1/m6A_Nanopore/control.zymo/guppy_result/extracted_filt",exist_ok = True)
chunk_out_f = "/home/heavens/twilight_hdd1/m6A_Nanopore/control.zymo/guppy_result/extracted_filt/chunks.npy"
seq_out_f = "/home/heavens/twilight_hdd1/m6A_Nanopore/control.zymo/guppy_result/extracted_filt/references.npy"
seq_len_out_f = "/home/heavens/twilight_hdd1/m6A_Nanopore/control.zymo/guppy_result/extracted_filt/reference_lengths.npy"
np.save(seq_out_f,seq_filt)
np.save(seq_len_out_f,seq_len_filt)
np.save(chunk_out_f,chunk[mask])