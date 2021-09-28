"""
Created on Wed Apr 28 23:03:36 2021

@author: Haotian Teng
"""

import os
import sys
import h5py
import numpy as np
import argparse
import seaborn as sns
from tqdm import tqdm
from xron.utils.seq_op import fast5_iter,norm_by_noisiest_section
from xron.utils.align import MetricAligner
from Bio.Seq import Seq
alt_map = {'ins':'0','M':'A','U':'T'}
complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} 

def reverse_complement(seq):    
    return str(Seq(seq).reverse_complement())

def clean_repr(seq):
    for k,v in alt_map.items():
        seq = seq.replace(k,v)
    return seq

def retrive_seq(seq_h,event_stride):
    moves = np.asarray(seq_h['BaseCalled_template']['Move'])
    try:
        seq = np.asarray(seq_h['BaseCalled_template']['Fastq']).tobytes().decode('utf-8').split('\n')[1]
    except UnicodeDecodeError:
        seq = str(np.asarray(seq_h['BaseCalled_template']['Fastq']).astype(str)).split('\n')[1]
    pos = np.repeat(np.cumsum(moves)-1,repeats = event_stride).astype(np.int32)
    return seq,pos
    

def extract(args):
    iterator = fast5_iter(args.input_fast5,mode = 'r')
    if args.extract_seq:
        print("Read reference genome.")
        aligner = MetricAligner(args.reference,options = '-x ont2d')
    print("Begin process the reads.")
    meta_info = []
    chunks = []
    seqs = []
    meds = []
    mads = []
    if args.mode == "rna" or args.mode == "rna_meth":
        reverse_sig = True
        
    for read_h,signal,fast5_f,read_id in tqdm(iterator):
        read_len = len(signal)
        signal,med,mad = norm_by_noisiest_section(signal)
        signal = signal.astype(np.float16)
        meds.append(med)
        mads.append(mad)
        if reverse_sig:
            signal = signal[::-1]
        if args.extract_seq: 
            seq, pos = retrive_seq(read_h['Analyses/Basecall_1D_%s'%(args.basecall_entry)],
                                   args.stride)
            seq = clean_repr(seq) #M->A, U->T
            hits,ref_seq,ref_idx = aligner.ref_seq(seq)
            if not hits:
                continue
            assert np.all(np.diff(ref_idx)>=0)
            start = int(read_h['Analyses/Segmentation_%s/Summary/segmentation'%(args.basecall_entry)].attrs['first_sample_template'])
            if reverse_sig:
                signal = signal[:-start]
                pos = pos[::-1]
                pos = pos[0] - pos 
            else:
                signal = signal[start:]
            signal = signal[:len(pos)]
            if len(signal) == 0:
                continue
            read_len = len(pos)
            # basecall_seq = seq
            for x in np.arange(0,read_len,args.chunk_len):
                s,e = pos[x:x+args.chunk_len][[0,-1]]
                mask = (ref_idx>=s)&(ref_idx<=e)
                if sum(mask) > 0:
                    r_s,r_e = np.where(mask)[0][[0,-1]]
                    # print("Basecall sequence:%s"%(basecall_seq[s:e+1]))
                    seq = ref_seq[r_s:r_e+1]
                    # print("Aligned seqeuence:%s"%(seq))
                    if args.mode == 'rna-meth':
                        seq.replace('A','M')
                    seqs.append(seq)
                else:
                    seqs.append('')
        current_chunks = np.split(signal,np.arange(0,read_len,args.chunk_len))[1:]
        last_chunk = current_chunks[-1]
        current_chunks[-1]= np.pad(last_chunk,(0,args.chunk_len-len(last_chunk)),'constant',constant_values = (0,0))
        chunks += current_chunks
        meta_info += [(fast5_f,read_id,str(args.chunk_len),str(args.stride))]*len(current_chunks)
        if args.max_n and len(chunks)>args.max_n:
            chunks = chunks[:args.max_n]
            seqs = seqs[:args.max_n]
            meta_info = meta_info[:args.max_n]
            break
    chunks = np.stack(chunks,axis = 0)
    np.savetxt(os.path.join(args.output,'meta.csv'),meta_info,fmt="%s")
    np.save(os.path.join(args.output,'chunks.npy'),chunks)
    print("Average median value %f"%(np.mean(meds)))
    print("Average median absolute deviation %f"%(np.mean(mads)))
    np.save(os.path.join(args.output,'mm.npy'),(mads,meds))
    if args.extract_seq:
        seq_lens = [len(i) for i in seqs]
        seqs = np.array(seqs)
        seq_lens = np.array(seq_lens)
        np.save(os.path.join(args.output,'seqs.npy'),seqs)
        np.save(os.path.join(args.output,'seq_lens.npy'),seq_lens)

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
    parser.add_argument('--chunk_len',
                        default = 4000,
                        type=int,
                        help="The lenght of the segment in chunk.")
    parser.add_argument('--max_n',
                        default = None,
                        type=int,
                        help="The maximum number of the segments to be extracted")
    parser.add_argument("--extract_seq",
                        action = "store_true",  
                        help = "If the sequence information is going to be\
                            extracted.")
    parser.add_argument('--basecall_entry',
                        default = "000",
                        help="The entry number in /Analysis/ to look into, for\
                            example 000 means looking for Basecall_1D_000.")
    parser.add_argument('--stride',
                        default = 10,
                        type = int,
                        help = "The length of stride used in basecall model,\
                        for guppy RNA fast, this number is 12, for guppy RNA\
                        hac model, this number is 10, for xron this number is\
                        5."
                        )
    parser.add_argument('--reference',
                        default = None,
                        help = "The reference genome, it's required when\
                        extract_seq is set to True")
    parser.add_argument('--mode',
                        default = "rna",
                        help = "Can be one of this mode: rna, dna, rna-meth")
    FLAGS = parser.parse_args(sys.argv[1:])
    if FLAGS.extract_seq:
        if not FLAGS.reference:
            raise ValueError("Reference genome is required when extract the \
                             sequence.")
    os.makedirs(FLAGS.output,exist_ok = True)
    extract(FLAGS)


