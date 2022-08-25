"""
Created on Wed Jul 20 00:42:56 2022

@author: Haotian Teng
"""
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


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
                    
def m6A_ratio(sequences):
    M_count = [x.count("M") for x in sequences]                    
    A_count = [x.count("A") for x in sequences]
    return np.sum(M_count)/(np.sum(A_count) + np.sum(M_count))
                

if __name__ == "__main__":
    scratch = "/home/heavens/bridge_scratch/"
    m6A_fs = [scratch +"NA12878_RNA_IVT/xron_retrain/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_25_pct/20210430_1751_X2_FAP66339_8447fb8b/xron_retrain25pctRandom/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_50_pct/20210430_1751_X3_FAQ16600_fe8f7999/xron_retrain25pctRandom/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_75_pct/20210430_1745_X1_FAQ15457_c865db38/xron_retrain25pctRandom/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_90_pct/20210430_1745_X2_FAQ15454_23428362/xron_retrain25pctRandom/fastqs"]
    # control_f = [scratch + ""]
    records = [read_fastqs(x) for x in tqdm(m6A_fs)]
    ratios = [m6A_ratio(x['sequences']) for x in tqdm(records)]
    x = [0,0.25,0.50,0.75,0.90]
    plt.plot(x,ratios,'.')
    plt.plot(x,x,'.')