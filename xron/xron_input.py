# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Mon Mar 27 14:04:57 2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import sys
import tempfile

import h5py
import numpy as np
from statsmodels import robust
from six.moves import range
from six.moves import zip
import torch
from xron.utils import progress
from xron import __version__
from packaging import version
SIGNAL_DTYPE=np.int16
raw_labels = collections.namedtuple('raw_labels', ['start', 'length', 'base'])
MIN_LABEL_LENGTH = 2
MIN_SIGNAL_PRO = 0.3
MEDIAN=0
MEAN=1

class dataset()


def read_signal_fast5(fast5_path, normalize=None):
    """
    Read signal from the fast5 file.
    TODO: To make it compatible with PromethION platform.
    """
    root = h5py.File(fast5_path, 'r')
    signal = np.asarray(list(root['/Raw/Reads'].values())[0][('Signal')])
    uniq_arr=np.unique(signal)
    if len(signal) == 0:
        return signal.tolist()
    if normalize == MEAN:
        signal = (signal - np.mean(uniq_arr)) / np.float(np.std(uniq_arr))
    elif normalize == MEDIAN:
        signal = (signal - np.median(uniq_arr)) / np.float(robust.mad(uniq_arr))
    return signal.tolist()
    
def read_label(file_path, skip_start=10, window_n=0):
    f_h = open(file_path, 'r')
    start = list()
    length = list()
    base = list()
    all_base = list()
    if skip_start < window_n:
        skip_start = window_n
    for line in f_h:
        record = line.split()
        all_base.append(base2ind(record[2]))
    f_h.seek(0, 0)  # Back to the start
    file_len = len(all_base)
    for count, line in enumerate(f_h):
        record = line.split()   
        if count < skip_start or count > (file_len - skip_start - 1):
            continue
        start.append(int(record[0]))
        length.append(int(record[1]) - int(record[0]))
        k_mer = 0
        for i in range(window_n * 2 + 1):
            k_mer = k_mer * 4 + all_base[count + i - window_n]
        base.append(k_mer)
    return raw_labels(start=start, length=length, base=base)


def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor
    """
    values = []
    indices = []
    for batch_i, label_list in enumerate(label_batch[:, 0]):
        for indx, label in enumerate(label_list):
            if indx >= label_batch[batch_i, 1]:
                break
            indices.append([batch_i, indx])
            values.append(label)
    shape = [len(label_batch), max(label_batch[:, 1])]
    return indices, values, shape


def base2ind(base, alphabet_n=4, base_n=1):
    """base to 1-hot vector,
    Input Args:
        base: current base,can be AGCT, or AGCTX for methylation.
        alphabet_n: can be 4 or 5, related to normal DNA or methylation call.
        """
    if alphabet_n == 4:
        Alphabeta = ['A', 'C', 'G', 'T']
        alphabeta = ['a', 'c', 'g', 't']
    elif alphabet_n == 5:
        Alphabeta = ['A', 'C', 'G', 'T', 'X']
        alphabeta = ['a', 'c', 'g', 't', 'x']
    else:
        raise ValueError('Alphabet number should be 4 or 5.')
    if base.isdigit():
        return int(base) / 256
    if ord(base) < 97:
        return Alphabeta.index(base)
    else:
        return alphabeta.index(base)
    #

def test_xron_dummy_input():
    DATA_FORMAT = np.dtype([('start','<i4'),
                            ('length','<i4'),
                            ('base','S1')]) 
    ### Generate dummy dataset and check input ###
    dummy_dir = './Dummy_data/'
    if not os.path.isdir(dummy_dir):
        os.makedirs(dummy_dir)
    dummy_fast5 = os.path.join(dummy_dir,'fast5s')
    dummy_raw = os.path.join(dummy_dir,'raw')
    if not os.path.isdir(dummy_fast5):
        os.makedirs(dummy_fast5)
    file_num = 10
    base_signal = {'A':100,'C':200,'G':300,'T':400}
    bases = ['A','C','G','T']
    for i in range(file_num):
        file_n = os.path.join(dummy_fast5,'dummy_' + str(i) + '.fast5')
        length = np.random.randint(40000,50000)
        start = 0
        start_list = []
        length_list = []
        base_list = []
        raw_signal = []
        while start < length-1:
            start_list.append(start)
            step = min(length-start-1, np.random.randint(5,150))
            length_list.append(step)
            start = start + step
            base = bases[np.random.randint(len(bases))]
            base_list.append(base)
            raw_signal = raw_signal + [base_signal[base]] + [base_signal[base]-1]*(step-1)
        event_matrix = np.asarray(list(zip(start_list,length_list,base_list)),dtype = DATA_FORMAT)
        with h5py.File(file_n,'w') as root:
            if '/Raw' in root:
                del root['/Raw']
            raw_h = root.create_dataset('/Raw/Reads/Read_'+ str(i)+'/Signal',
                                        shape = (len(raw_signal),),
                                        dtype = np.int16)
            channel_h=root.create_dataset('/UniqueGlobalKey/channel_id/',shape=[],dtype=np.int16)
            channel_h.attrs['offset']=0
            channel_h.attrs['range']=1
            channel_h.attrs['digitisation']=1
            raw_h[...] = raw_signal[::-1]
            if '/Analyses' in root:
                del root['/Analyses']
            event_h = root.create_dataset('/Analyses/Corrected_000/BaseCalled_template/Events', 
                                          shape = (len(event_matrix),),
                                          maxshape=(None,),
                                          dtype = DATA_FORMAT)
            event_h[...] = event_matrix
            event_h.attrs['read_start_rel_to_raw'] = 0
            
    class Args(object):
        def __init__(self):
            self.input = dummy_fast5
            self.output = dummy_raw
            self.basecall_group = 'Corrected_000'
            self.mode = 'rna'
            self.batch = 1
            self.basecall_subgroup = 'BaseCalled_template'
            self.unit=True
            self.min_bps = 0
            self.n_errors = 5
    from xron.utils import raw
    args = Args()
    raw.run(args)
    train = read_raw_data_sets(dummy_raw,seq_length=1000,h5py_file_path=os.path.join(dummy_dir,'cache.fast5'))
    
    for i in range(100):
        inputX, sequence_length, label = train.next_batch(10,shuffle=False)
        accum_len = 0
        for idx,x in enumerate(inputX):
            x = inputX[idx][:sequence_length[idx]]
            y = list()
            for x_idx, signal in enumerate(x):
                if x_idx==0:
                    y.append(signal)
                else:
                    if (abs(signal - x[x_idx-1]) >0.1) or (signal - x[x_idx-1])>0:
                        y.append(signal)
            corr = np.corrcoef(y, label[1][accum_len:accum_len + len(y)])[0, 1]
            for loc in label[0][accum_len:accum_len + len(y)]:
                assert(loc[0] == idx)
            accum_len += len(y)
            assert abs(corr - 1)< 1e-6
    print("Input pipeline dummy data test passed!")
                    
#
if __name__ == '__main__':
    test_xron_dummy_input()
#    TEST_DIR='/home/heavens/Documents/test/'
#    train = read_tfrecord(TEST_DIR,"train.tfrecords",seq_length=1000,h5py_file_path=os.path.join(TEST_DIR,'cache.fast5'))
