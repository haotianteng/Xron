"""
Created on Thu Sep 16 13:34:47 2021

@author: Haotian Teng
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
fig,axs = plt.subplots()
print("Load NA12878 control dataset.")
chunk_NA = np.load("/home/heavens/bridge_scratch/NA12878_RNA_IVT/xron_output/kmers_xron_4000_noise/chunks.npy")
print("Load 100pct m6A dataset")
chunk_m6A = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/kmers_xron_4000_noise/chunks.npy")
print("Load renormalized 100pct m6A IVT dataset.")
chunk_m6A_renorm = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/kmers_xron_4000_noise/chunks_renorm_2.npy")
sns.distplot(np.mean(chunk_NA,axis = 1),ax = axs,label = "NA12878 Control",hist = None)
sns.distplot(np.mean(chunk_m6A,axis = 1),ax = axs, label = "m6A",hist = None)
sns.distplot(np.mean(chunk_m6A_renorm,axis = 1),ax = axs, label = "m6A_renorm",hist = None)
axs.set_xlim(left = -2.5,right = 2.5)
fig.legend()