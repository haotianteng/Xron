"""
Created on Wed Apr 28 23:03:36 2021

@author: Haotian Teng
"""

import os
import sys
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from scipy.signal import find_peaks
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

def fast5_iter(fast5_dir,mode = 'r'):
    for (dirpath, dirnames, filenames) in os.walk(fast5_dir+'/'):
        for filename in filenames:
            if not filename.endswith('fast5'):
                continue
            abs_path = os.path.join(dirpath,filename)
            try:
                root = h5py.File(abs_path,mode = mode)
            except OSError as e:
                print("Reading %s failed due to %s."%(abs_path,e))
                continue
            for read_id in root:
                read_h = root[read_id]['Raw']
                signal = np.asarray(read_h[('Signal')],dtype = np.float32)
                read_id = read_h.attrs['read_id']
                yield read_h,signal,abs_path,read_id.decode("utf-8")

def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    Same method used in Bonito Basecaller.
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def norm_by_noisiest_section(signal, samples=100, threshold=6.0):
    """
    Normalise using the medmad from the longest continuous region where the
    noise is above some threshold relative to the std of the full signal.This
    function is borrowed from Bonito Basecaller.
    """
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0; noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))

    if len(peaks):
        widest = np.argmax(info['widths'])
        med, mad = med_mad(signal[info['left_bases'][widest]: info['right_bases'][widest]])
    else:
        med, mad = med_mad(signal)
    return (signal - med) / mad

def retrive_seq(seq_h,event_stride):
    moves = np.asarray(seq_h['BaseCalled_template']['Move'])
    seq = np.asarray(seq_h['BaseCalled_template']['Fastq']).tobytes().decode('utf-8').split('\n')[1]
    pos = np.repeat(np.cumsum(moves)-1,repeats = event_stride)
    return seq,pos
    

def extract(args):
    iterator = fast5_iter(args.input_fast5,mode = 'r')
    aligner = MetricAligner(args.reference,options = '-x ont2d')
    meta_info = []
    chunks = []
    seqs = []
    for read_h,signal,fast5_f,read_id in tqdm(iterator):
        read_len = len(signal)
        signal = norm_by_noisiest_section(signal).astype(np.float16)
        if args.mode == "rna" or args.mode == "rna-meth":
            signal = signal[::-1]
        if args.extract_seq:
            seq, pos = retrive_seq(read_h['Analyses/Basecall_1D_%s'%(args.basecall_entry)],
                                   args.stride)
            seq = clean_repr(seq) #M->A, U->T
            hit = aligner.align_seq(seq)
            if not hit:
                continue
            hit = hit[0]
            if hit.orient == '-':
                seq = reverse_complement(seq)
                
            start = read_h['Analyses/Segmentation_001/Summary/segmentation'].attrs['first_sample_template']
            signal = signal[start:]
            read_len -= start
            for x in np.arange(0,read_len,args.chunk_len):
                s,e = pos[x:x+args.chunk_len][[1,-1]]
                seqs.append(seq[s:e])
        current_chunks = np.split(signal,np.arange(0,read_len,args.chunk_len))[1:]
        last_chunk = current_chunks[-1]
        current_chunks[-1]= np.pad(last_chunk,(0,args.chunk_len-len(last_chunk)),'constant',constant_values = (0,0))
        chunks += current_chunks
        meta_info += [(fast5_f,read_id)]*len(current_chunks)
    chunks = np.stack(chunks,axis = 0)
    np.savetxt(os.path.join(args.output,'meta.csv'),meta_info,fmt="%s")
    np.save(os.path.join(args.output,'chunks.npy'),chunks)
    if args.extract_seq:
        seq_lens = [len(i) for i in seqs]
        pad = max(seq_lens)
        seq_chunks = np.array([np.pad(i, ((0,pad-j),(0,0))) for i,j in zip(seqs,seq_lens)])
        seq_lens = np.array(seq_lens)
        np.save(os.path.join(args.output,'seqs.npy'),seq_chunks)
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
    parser.add_argument("--extract_seq",
                        action = "store_true",
                        help = "If the sequence information is going to be\
                            extracted.")
    parser.add_argument('--basecall_entry',
                        default = "000",
                        help="The entry number in /Analysis/ to look into, for\
                            example 000 means looking for BaseCall_1D_000.")
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
    extract(FLAGS)
