#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 19:55:38 2022

@author: haotian
"""

import os
import sys
import toml
import numpy as np
import argparse

def merge(args):
    fs = args.fs
    keys = args.keys
    collections = {x+'.npy':[] for x in keys}
    for f_i,sub_f in enumerate(fs):
        print("Collecting from %s..."%(sub_f))
        npy_files = [x for x in os.listdir(sub_f) if x.endswith(".npy")]
        for key in collections.keys():
            if key not in npy_files:
                print("The chunk %s is not in %s, stop collecting it."%(key,sub_f))
                del collections[key]
                continue
            else:
                print("Collecting %s..."%(key))
                pieces = np.load(os.path.join(sub_f,key),mmap_mode = "r")
                print("Collected %d instances"%(len(pieces)))
                if args.max is not None and len(pieces) > args.max[f_i]:
                    print("Thresholding it to %d instances"%(args.max[f_i]))
                    pieces = pieces[:args.max[f_i]]
                collections[key].append(pieces)
    shapes = [sum([len(y) for y in x]) for x in collections.values()]
    assert len(np.unique(shapes)) == 1

    for key in collections.keys():
        print("merge %s"%(key))
        if collections[key][0].ndim > 1:
            np.save(os.path.join(args.output,key),np.vstack(tuple(collections[key])))
        else:
            np.save(os.path.join(args.output,key),np.hstack(tuple(collections[key])))

def main(args):
    merge(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training RHMM model')
    parser.add_argument("-i","--input", type = str, required = True,
                        help = "Data folder contains npy file, separated by comma.")
    parser.add_argument("-o","--output",required = True, type = str,
                        help = "The output folder to store the merged dataset.")
    parser.add_argument("-k","--key", type = str, default = "chunks,path,seqs,seq_lens,durations",
                        help = "The name of npy items need to be collected, separated by comma.")
    parser.add_argument("-m","--max", default = None,
                        help = "The maximum number of instances to be include in each dataset, can be a list of int separated by commas specify maximum number for each datasets.")
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.output,exist_ok = True)
    args.fs = args.input.strip().split(',')
    args.keys = args.key.strip().split(',')
    if ',' in args.max:
        args.max = [int(x) for x in args.max.split(',')]
    else:
        args.max = [int(args.max)] * len(args.fs)
    assert len(args.fs) == len(args.max)
    main(args)
