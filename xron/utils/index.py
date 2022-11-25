#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:47:48 2022

@author: heavens
"""
import os
import h5py
import argparse
from xron.utils.seq_op import fast5_shallow_iter
from xron.utils.fastIO import read_fastq

def indexing(args):
    fastq_records = read_fastq(args.fastq)
    fast5_records = {}
    print("Indexing fastq files.")
    with open(args.fastq+'.index', 'w+') as f:
        for root,abs_path in fast5_shallow_iter(args.fast5,tqdm_bar = True):
            read_ids = list(root.keys())
            for id in read_ids:
                fast5_records[id[5:]] = abs_path #read_id is like "read_00000000-0000-0000-0000-0000000000"
        for id in fastq_records['name']:
            if id in fast5_records.keys():
                f.write(id+'\t'+fast5_records[id]+'\n')
            else:
                raise KeyError('fastq readid %s not found in fast5'%(id))
    print("Indexing fastq file has been finished.")            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fast5', required = True, type=str, help='folder that contains fast5 output')
    parser.add_argument('--fastq', required = True, type=str, help='The merged fastq file')
    args = parser.parse_args()
    indexing(args)