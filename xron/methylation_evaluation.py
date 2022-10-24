"""
Created on Wed Jul 20 00:42:56 2022

@author: Haotian Teng
"""
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from xron.utils.fastIO import read_fastqs
import seaborn as sns
from matplotlib import pyplot as plt

def m6A_ratio(sequences,reduced_sum = True):
    M_count = [x.count("M") for x in sequences]                    
    A_count = [x.count("A") for x in sequences]
    if reduced_sum:
        return np.sum(M_count)/(np.sum(A_count) + np.sum(M_count))
    else:
        return np.asarray(M_count)/(np.asarray(A_count) + np.asarray(M_count))
                       
if __name__ == "__main__":
    scratch = "/home/heavens/bridge_scratch/"
    
    x = [0,0.25,0.50,0.75,0.90,1.0]
    m6A_fs = ["/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/xron_crosslink/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_25_pct/20210430_1751_X2_FAP66339_8447fb8b/xron_crosslink/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_50_pct/20210430_1751_X3_FAQ16600_fe8f7999/xron_crosslink/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_75_pct/20210430_1745_X1_FAQ15457_c865db38/xron_crosslink/fastqs",
              "/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_90_pct/20210430_1745_X2_FAQ15454_23428362/xron_crosslink/fastqs",
              "/home/heavens/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/xron_crosslink/fastqs"]
    
    x = [0,1.0]
    m6A_fs = ["/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/xron_crosslink_finetune/fastqs",
              "/home/heavens/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/xron_crosslink_finetune/fastqs"]
    
    x = [0,0.3]
    m6A_fs = ["/home/heavens/bridge_scratch/ime4_Yearst/Yearst/ko_raw_fast5/xron_crosslink/fastqs",
              "/home/heavens/bridge_scratch/ime4_Yearst/Yearst/wt_raw_fast5/xron_crosslink/fastqs"]
    
    records = [read_fastqs(x) for x in tqdm(m6A_fs)]
    ratios = [m6A_ratio(x['sequences'],reduced_sum = False) for x in tqdm(records)]
    
    mix_prop = [[xi]*len(ratios[i]) for i,xi in enumerate(x)]
    ratios = np.concatenate(ratios,axis = 0)
    mix_prop = np.concatenate(mix_prop,axis = 0)
    df = pd.DataFrame({"basecall_ratio": ratios,"prepare_ratio": mix_prop})
    
    fig, axes = plt.subplots(figsize=(5,5))
    # sns.violinplot(data = df,x = "mix_prop",y = "ratios",showmeans = True,showmedians = True)
    sns.boxplot(data = df,x = "prepare_ratio",y = "basecall_ratio",showfliers = False,ax = axes)
    axes.set_xlabel("m6A propertion during IVT")
    axes.set_ylabel("Basecalled m6A ratio")
    fig.savefig("/home/heavens/bridge_scratch/Xron_Project/methylation_ratio.png")
