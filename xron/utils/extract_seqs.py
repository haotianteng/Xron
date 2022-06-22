"""
Created on Wed Jan 26 12:55:10 2022

@author: Haotian Teng
"""
import os,sys,argparse
from tqdm import tqdm
import h5py
import numpy as np
from xron.utils.seq_op import fast5_iter
from pathlib import Path


def retrive_fastq(seq_h):
    try:
        seq = np.asarray(seq_h['BaseCalled_template']['Fastq']).tobytes().decode('utf-8')
    except:
        seq = str(np.asarray(seq_h['BaseCalled_template']['Fastq']).astype(str))
    return seq

def extract(args):
    iterator = fast5_iter(args.input_fast5,mode = 'r')
    for read_h,signal,fast5_f,read_id in tqdm(iterator):
        seq = retrive_fastq(read_h['Analyses/Basecall_1D_%s'%(args.basecall_entry)])
        with open(os.path.join(args.output,Path(fast5_f).stem+'.fastq'),'a+') as f:
            f.write(seq)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='xron',   
                                     description='A Unsupervised Nanopore basecaller.')
    parser.add_argument('-i', 
                        '--input_fast5', 
                        required = True,
                        help="File path or Folder path to the fast5 file.")
    parser.add_argument('-o',
                        '--output',
                        required = True,
                        help="Output folder.")
    parser.add_argument('--basecall_entry',
                        default = "000",
                        help="The entry number in /Analysis/ to look into, for\
                            example 000 means looking for Basecall_1D_000.")
                            
    FLAGS = parser.parse_args(sys.argv[1:])
    os.makedirs(FLAGS.output,exist_ok = True)
    extract(FLAGS)