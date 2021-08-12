"""
Created on Wed Aug 11 21:32:48 2021

@author: Haotian Teng
"""

import h5py
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def retrive_seq(seq_h,event_stride):
    moves = np.asarray(seq_h['BaseCalled_template']['Move'])
    seq = np.asarray(seq_h['BaseCalled_template']['Fastq']).tobytes().decode('utf-8').split('\n')[1]
    pos = np.repeat(np.cumsum(moves)-1,repeats = event_stride)
    return seq,pos

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

def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    Same method used in Bonito Basecaller.
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad
                
test_file = "/home/heavens/Documents/FAL28748_f879e763ca81503aa9fc726c3b2d1169b1c9aed8_0.fast5"
read_id = "read_000355dc-6d59-40a9-bdca-5414b3885b51"
# template_f = 
show_n = 2000
with h5py.File(test_file,mode = 'r') as root:
    read_h = root[read_id]
    signal_h = read_h['Raw']
    signal = np.asarray(signal_h[('Signal')],dtype = np.float32)
    start = int(read_h['Analyses/Segmentation_001/Summary/segmentation'].attrs['first_sample_template'])
    signal = signal[start:]
    seq, pos = retrive_seq(read_h['Analyses/Basecall_1D_001'],12)
    figs,axs = plt.subplots(nrows = 2)
    axs[0].plot(signal[:show_n])
    axs[1].plot(pos[:show_n])