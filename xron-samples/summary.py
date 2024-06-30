#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:38:12 2022

@author: heavens
"""
import os
import sys
import argparse
import numpy as np
import seaborn as sns
from operator import mod
from matplotlib import pyplot as plt
from xron.utils.fastIO import read_entry
from xron.utils.plot_op import auc_plot
from xron.utils.seq_op import fast5_iter

def AUC(TP,FP):
    """Calculate Area under curve given the true positive and false positive
    array
    """
    TP = TP[::-1] if TP[0]>TP[-1] else TP
    FP = FP[::-1] if FP[0]>FP[-1] else FP
    TP = [0] + TP if TP[0] != 0 else TP
    TP = TP + [1] if TP[-1] != 1 else TP
    FP = [0] + FP if FP[0] != 0 else FP
    FP = FP + [1] if FP[-1] != 1 else FP
    FP = np.asarray(FP)
    TP = np.asarray(TP)
    return np.sum((TP[1:] + TP[:-1])*(FP[1:]-FP[:-1])/2)
    
def posterior_decode(posterior,
                     M_threshold:float = 0.5):
    """Decode the posterior probability to get the modified ratio"""
    called = posterior > M_threshold
    return called.sum()/called.size

def run(args):
    control_fast5 = args.control
    modified_fast5 = args.positive
    output = args.output
    label = "" if args.label is None else "_"+args.label
    n_total = np.inf if args.max_n == -1 else args.max_n
    TP,FP = [],[]
    c_p,m_p = [],[]
    for read_h,signal,abs_path,read_id in fast5_iter(control_fast5,mode = 'r',tqdm_bar = True):
        try:
            p = read_entry(read_h,entry = "ModifiedProbability",index = "001")
            c_p.append(p)
        except:
            pass
        if len(c_p) >= n_total:
            break
    for read_h,signal,abs_path,read_id in fast5_iter(modified_fast5,mode = 'r',tqdm_bar = True):
        try:
            p = read_entry(read_h,entry = "ModifiedProbability",index = "001")
            m_p.append(p)
        except:
            pass
        if len(m_p) >= n_total:
            break
    c_p = np.hstack(c_p)
    m_p = np.hstack(m_p)
    for t in np.arange(0,1.0001,0.002):
        TP.append(posterior_decode(m_p,M_threshold = t))
        FP.append(posterior_decode(c_p,M_threshold = t))
    fig,axs = plt.subplots(figsize = (5,5))
    auc_plot(TP,FP,axs = axs)
    fig.savefig(os.path.join(output,"roc%s.png"%(label)))
    np.save(os.path.join(output,"TP%s.npy"%(label)),TP)
    np.save(os.path.join(output,"FP%s.npy"%(label)),FP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--control",type = str, required = True,help = "Control fast5 file")
    parser.add_argument("-p","--positive",type = str,required = True,help = "Positive fast5 file")
    parser.add_argument("-o","--output",type = str,required = True, help = "Output directory")
    parser.add_argument("-l","--label",type = str,help = "Label for the output file")
    parser.add_argument("-n","--max_n",type = int,default = -1,help = "Maximum number of reads to use")
    args = parser.parse_args()
    run(args)