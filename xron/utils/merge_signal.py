"""
Created on Tue Aug  3 23:24:39 2021

@author: Haotian Teng
"""
import numpy as np
import os
import sys
import argparse


def main(args):
    chunks = []
    counts = []
    for i in args.input:
        try:
            c = np.load(i)
        except FileNotFoundError:
            print("Didn't find %s, skip."%(i))
            continue
        counts.append(len(c))
        chunks.append(c)
        print("Read %d chunks from %s"%(len(c),i))
    if args.equal_size:
        min_size = min(counts)
        chunks = [x[:min_size] for x in chunks]
    chunk_all = np.concatenate(chunks,axis = 0)
    np.save(os.path.join(args.output,'chunks_all.npy'),chunk_all)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge signal chunks from different folder.')
    parser.add_argument('-i', '--input', default = None,
                        help = "The input signal chunks, separate by comma.")
    parser.add_argument('-o', '--output', required = True,
                        help = "The output folder of the merged dataset.")
    parser.add_argument('--equal_size',default = False, type = bool,
                        help = "If make the size from each chunks equal.")
    args = parser.parse_args(sys.argv[1:])
    args.input = args.input.split(',')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)