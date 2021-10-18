"""
Created on Mon Apr 19 13:59:01 2021

@author: Haotian Teng
"""
from numpy import ndarray
import os
import h5py
import numpy as np
from typing import List
from scipy.signal import find_peaks

def raw2seq(raw:ndarray, blank_symbol:int = 0)->List[ndarray]:
    """
    Transfer raw ctc sequence by removing the consecutive repeat and blank symbol.

    Parameters
    ----------
    raw : ndarray
        The [N,C,L] raw ctc sequence, where N is the batch size and L is the 
        sequence length.
    blank_symbol : int, optional
        The index of the blank symbol. The default is 0.

    Returns
    -------
    List[ndarray]
        Length N list contains Transfered sequences.

    """
    mask = np.diff(raw).astype(bool)
    mask = np.insert(mask,0,True,axis = 1)
    moves = np.logical_and(raw!=blank_symbol,mask)
    np.ma.array(raw,mask = mask).tolist()
    seqs = np.ma.array(raw,mask = np.logical_not(mask)).tolist(None)
    out_seqs = [[x-1 for x in seq if x is not None and x!=blank_symbol] for seq in seqs]
    return out_seqs,moves


def list2string(input_v, base_type):
    if base_type == 0:
        base_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'b'}
    if base_type == 1:
        base_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    if base_type == 2:
        base_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'M'}
    if isinstance(base_type,dict):
        base_dict = base_type
    return "".join([base_dict[item] for item in input_v])


def string2list(input_v, base_type):
    if base_type == 0:
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    if isinstance(base_type,dict):
        base_dict = base_type
    result = list()
    for item in input_v:
        result.append(base_dict[item])
    return result


def fast5_iter(fast5_dir,mode = 'r'):
    for (dirpath, dirnames, filenames) in os.walk(fast5_dir+'/'):
        for filename in filenames:
            if not filename.endswith('fast5'):
                continue
            abs_path = os.path.join(dirpath,filename)
            try:
                root = h5py.File(abs_path,mode = mode)
            except Exception as e:
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


def norm_by_noisiest_section(signal, samples=100, threshold=6.0,offset = 0.0):
    """
    Normalise using the medmad from the longest continuous region where the
    noise is above some threshold relative to the std of the full signal.This
    function is borrowed from Bonito Basecaller.

    Parameters
    ----------
    signal : 1-D np.array
        A 1D numpy array contain signal from one read.
    samples : Int, optional
        The window size. The default is 100.
    threshold : Float, optional
        Signal with std over threshold are used to find the peaks. The default is 6.0.
    offset : Float, optional
        The offset to apply to the signal. The default is 0.0.

    Returns
    -------
    normalized_signal: 1-D array
        The normalized signal.
    med : Float
        The median value.
    mad : Float
        The MAD value estimate the deviation of the signal.

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
    return (signal - med + offset) / mad, med, mad

def diff_norm_by_noisiest_section(signal, samples=100, threshold=6.0,offset = 0.0):
    """
    Calculate the difference of the signal S[1:]-S[:-1], scale the
    diff signal with the MAD value that extracted with the same method as
    norm_by_noisiest_section function.

    Parameters
    ----------
    signal : 1-D np.array
        A 1D numpy array contain signal from one read.
    samples : Int, optional
        The window size. The default is 100.
    threshold : Float, optional
        Signal with std over threshold are used to find the peaks. The default is 6.0.
    offset : Float, optional
        The offset to apply to the signal. The default is 0.0.

    Returns
    -------
    normalized_signal: 1-D array
        The differentiable normalized signal.
    med : Float
        The median value.
    mad : Float
        The MAD value estimate the deviation of the signal.

    """
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0; noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))
    diff_signal = signal[1:] - signal[:-1]
    if len(peaks):
        widest = np.argmax(info['widths'])
        med, mad = med_mad(diff_signal[info['left_bases'][widest]: info['right_bases'][widest]])
    else:
        med, mad = med_mad(diff_signal)
    return (diff_signal + offset) / mad, med, mad
