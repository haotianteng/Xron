"""
Created on Mon Mar 28 23:54:38 2022

@author: Haotian Teng
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt
strides = 11
root = h5py.File("/home/heavens/Documents/UCSC_Run1_20180129_IVT_RNA_201.fast5",'r')
for read_id in root:
    if read_id == "read_40634cc4-ff9c-41f3-bd2c-901cb23c0179":
        plt.figure(figsize = (40,20))
        plt.plot(np.asarray(root[read_id]['Raw/Signal'])[::-1])
        read_h = root[read_id]
        basecalled = read_h['Analyses/Basecall_1D_000/BaseCalled_template']
        hidden = np.asarray(basecalled['Hidden'])
        signal = np.asarray(read_h['Raw/Signal'])
        move = basecalled['Move']
        position_sig = np.repeat(np.cumsum(move)-1,repeats = strides).astype(np.int32)[:len(signal)]
        segmentation = np.where(np.diff(position_sig))[0]
        
        plt.figure(figsize = (80,40))
        max_T = 40000
        plt.plot(signal[::-1][-(max_T+5000):],color = "red")
        plt.vlines(x = segmentation[segmentation>(segmentation[-1]-max_T)] - len(signal)+max_T, 
                   ymin = min(signal), 
                   ymax = max(signal), 
                   color = "green", 
                   linestyles = "dotted", 
                   linewidth = 0.01, 
                   antialiased=False)
        
        plt.figure(figsize = (80,40))
        max_T = 40000
        plt.plot(signal[:max_T],color = "red")
        segmentation = segmentation[-1] - segmentation[::-1]
        plt.vlines(x = segmentation[segmentation<max_T], 
                   ymin = min(signal), 
                   ymax = max(signal), 
                   color = "green", 
                   linestyles = "dotted", 
                   linewidth = 0.01, 
                   antialiased=False)