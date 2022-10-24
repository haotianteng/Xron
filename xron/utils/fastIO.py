#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:35:26 2022

@author: heavens
"""
import os
import numpy as np

def read_fastqs(fastq_f):
    records = {"sequences":[],"name":[],"quality":[]}
    for fastq in os.listdir(fastq_f):
        with open(os.path.join(fastq_f,fastq),'r') as f:
            for line in f:
                if line.startswith("@"):
                    records['name'].append(line.strip()[1:])
                    records['sequences'].append(next(f).strip())
                    assert next(f).strip() == "+" #skip the "+"
                    records['quality'].append(next(f).strip())
    return records

def read_fastq(fastq):
    records = {"sequences":[],"name":[],"quality":[]}
    with open(fastq,'r') as f:
        for line in f:
            if line.startswith("@"):
                records['name'].append(line.strip()[1:])
                records['sequences'].append(next(f).strip())
                assert next(f).strip() == "+" #skip the "+"
                records['quality'].append(next(f).strip())
    return records

def read_fast5(read_h,index = "000"):
    result_h = read_h['Analyses/Basecall_1D_%s/BaseCalled_template'%(index)]
    logits = result_h['Logits']
    move = result_h['Move']
    try:
        seq = str(np.asarray(result_h['Fastq']).astype(str)).split('\n')[1]
    except:
        seq = np.asarray(result_h['Fastq']).tobytes().decode('utf-8').split('\n')[1]
    return np.asarray(logits),np.asarray(move),seq

def read_entry(read_h,entry:str,index = "000"):
    """
    Read a entry given the name

    """
    result_h = read_h['Analyses/Basecall_1D_%s/BaseCalled_template'%(index)]
    return np.asarray(result_h[entry])