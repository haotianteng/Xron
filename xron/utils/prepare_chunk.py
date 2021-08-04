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
                read_h = root[read_id]
                signal_h = read_h['Raw']
                signal = np.asarray(signal_h[('Signal')],dtype = np.float32)
                read_id = signal_h.attrs['read_id']
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
    if args.extract_seq:
        print("Read reference genome.")
        aligner = MetricAligner(args.reference,options = '-x ont2d')
    print("Begin process the reads.")
    meta_info = []
    chunks = []
    seqs = []
    if args.mode == "rna" or args.mode == "rna_meth":
        reverse_sig = True
        
    for read_h,signal,fast5_f,read_id in tqdm(iterator):
        read_len = len(signal)
        signal = norm_by_noisiest_section(signal).astype(np.float16)
        if reverse_sig:
            signal = signal[::-1]
        if args.extract_seq: 
            seq, pos = retrive_seq(read_h['Analyses/Basecall_1D_%s'%(args.basecall_entry)],
                                   args.stride)
            seq = clean_repr(seq) #M->A, U->T
            hits,ref_seq,ref_idx = aligner.ref_seq(seq)
            if not hits:
                continue
            start = int(read_h['Analyses/Segmentation_%s/Summary/segmentation'%(args.basecall_entry)].attrs['first_sample_template'])
            if reverse_sig:
                signal = signal[:-start]
                pos = pos[::-1]
                pos = pos[0] - pos + 1
            else:
                signal = signal[start:]
            signal = signal[:len(pos)]
            if len(signal) == 0:
                continue
            read_len = len(pos)
            for x in np.arange(0,read_len,args.chunk_len):
                s,e = pos[x:x+args.chunk_len][[0,-1]]
                mask = (ref_idx>=s)&(ref_idx<=e)
                if sum(mask) > 0:
                    r_s,r_e = np.where(mask)[0][[0,-1]]
                    seq = ref_seq[r_s:r_e+1]
                    if args.mode == 'rna-meth':
                        seq.replace('A','M')
                    seqs.append(seq)
                else:
                    seqs.append('')
        current_chunks = np.split(signal,np.arange(0,read_len,args.chunk_len))[1:]
        last_chunk = current_chunks[-1]
        current_chunks[-1]= np.pad(last_chunk,(0,args.chunk_len-len(last_chunk)),'constant',constant_values = (0,0))
        chunks += current_chunks
        meta_info += [(fast5_f,read_id)]*len(current_chunks)
        if args.max_n and len(chunks)>args.max_n:
            chunks = chunks[:args.max_n]
            break
    chunks = np.stack(chunks,axis = 0)
    np.savetxt(os.path.join(args.output,'meta.csv'),meta_info,fmt="%s")
    np.save(os.path.join(args.output,'chunks.npy'),chunks)
    if args.extract_seq:
        seq_lens = [len(i) for i in seqs]
        pad = max(seq_lens)
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


