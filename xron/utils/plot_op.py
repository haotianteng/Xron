#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 07:34:49 2022

@author: heavens
"""
from matplotlib import pyplot as plt
import numpy as np

def auc_plot(TP,FP,axs,add_head = False,print_AUC = True,**kwargs):
    c = kwargs['color'] if 'color' in kwargs.keys() else None
    label = kwargs['label'] if 'color' in kwargs.keys() else None
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs.keys() else 1
    TP = np.asarray(TP)
    FP = np.asarray(FP)
    assert np.all(TP[1:] - TP[:-1]>=0)
    assert np.all(FP[1:] - FP[:-1]>=0)
    if add_head:
        TP = np.concatenate(([0],TP,[1]))
        FP = np.concatenate(([0],FP,[1]))
    axs.plot(FP[:2],TP[:2],color = c,label = label,linewidth = linewidth)
    axs.set_xlabel("False Positive")
    axs.set_ylabel("True Positive")
    axs.set_xlim([-0.05, 1.05])
    axs.set_ylim([-0.05, 1.05])
    if print_AUC:
        axs.text(x = 0.8,y = 0,s = "AUC = %.2f"%(AUC(TP,FP)))
    axs.plot([-0.05,1.05],[-0.05,1.05],color = 'grey')

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
    