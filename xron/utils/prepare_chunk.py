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
    noise is above some threshold relative to the std of the full signal.
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

def extract(args):
    iterator = fast5_iter(args.input_fast5,mode = 'r')
    meta_info = []
    chunks = []
    for read_h,signal,fast5_f,read_id in tqdm(iterator):
        read_len = len(signal)
        signal = norm_by_noisiest_section(signal).astype(np.float16)
        current_chunks = np.split(signal,np.arange(0,read_len,args.chunk_len))[1:]
        last_chunk = current_chunks[-1]
        current_chunks[-1]= np.pad(last_chunk,(0,args.chunk_len-len(last_chunk)),'constant',constant_values = (0,0))
        chunks += current_chunks
        meta_info += [(fast5_f,read_id)]*len(current_chunks)
    chunks = np.stack(chunks,axis = 0)
    np.savetxt(os.path.join(args.output,'meta.csv'),meta_info,fmt="%s")
    np.save(os.path.join(args.output,'chunks.npy'),chunks)

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
    FLAGS = parser.parse_args(sys.argv[1:])
    extract(FLAGS)
