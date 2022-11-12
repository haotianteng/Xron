#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 07:32:52 2022

@author: haotian teng
"""

import os
import sys
import toml
import numpy as np
import argparse

def split(args):
    collections = {}
    fs = os.listdir(args.input)
    for f in fs:
        if f.endswith(".npy"):
            try:
                collections[f] = np.load(os.path.join(args.input,f))
            except ValueError:
                continue
    s = np.array([len(x) for x in collections.values()])
    if not(np.all(s==s[0])):
        print("Warning the npy files inside the folder has different size.")
    for i in np.arange(0,max(s),args.batch_size):
        sub_f = os.path.join(args.input,args.prefix+str(i//args.batch_size))
        os.makedirs(sub_f,exist_ok=True)
        for f,data in collections.items():
            np.save(os.path.join(sub_f,f),data[i:i+args.batch_size])

def merge(args):
    fs = os.listdir(args.input)
    fs = [x for x in fs if args.prefix in x]
    i = 0
    collections = {}
    while args.prefix+str(i) in fs:
        sub_f = os.path.join(args.input,args.prefix + str(i))
        for f in os.listdir(sub_f):
            if f.endswith(".npy"):
                pieces = np.load(os.path.join(sub_f,f))
                if f in collections.keys():
                    collections[f].append(pieces)
                else:
                    collections[f] = [pieces]
        i+=1
    for key in collections.keys():
        print("merge %s"%(key))
        if collections[key][0].ndim > 1:
            np.save(os.path.join(args.input,key),np.vstack(tuple(collections[key])))
        else:
            np.save(os.path.join(args.input,key),np.hstack(tuple(collections[key])))

def main(args):
    if args.merge:
        merge(args)
    else:
        split(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training RHMM model')
    parser.add_argument("-i","--input", type = str, required = True,
                        help = "Data folder contains the chunk, kmer sequence.")
    parser.add_argument("-b","--batch_size", type = int, default = 4000,
                        help = "The batch size of each subfolder.")
    parser.add_argument("--prefix", type = str, default = "subdata",
                        help = "The prefix of the sub-folders.")
    parser.add_argument("--reverse",action = "store_true", dest = "merge",
                        help = "Reverse the split operation, surrogate the data.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
