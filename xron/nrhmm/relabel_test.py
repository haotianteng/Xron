"""
Created on Tue Jun 28 02:32:18 2022

@author: Haotian Teng
"""
import toml
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from xron.nrhmm.hmm_relabel import get_effective_kmers


chunk_control = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_xron_4000_noise/chunks.npy")
chunk_m6A = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/kmers_guppy_4000_dwell/chunks.npy")
kmer_m6A = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/kmers_guppy_4000_dwell/kmers.npy")
chunk_control = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_dwell/chunks.npy")
chunk_control_renorm = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_dwell/chunks_renorm.npy")
kmer_control = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_dwell/kmers.npy")
chunk_control_NA = np.load("/home/heavens/bridge_scratch/NA12878_RNA_IVT/guppy_train/kmers_guppy_4000_dwell/chunks.npy")
config = toml.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/kmers_guppy_4000_dwell/config.toml")
chunk_control_xron = np.load("/home/heavens/bridge_scratch/NA12878_RNA_IVT/xron_output/kmers_guppy_4000_dwell/chunks_renorm.npy")
chunk_m6A90 = np.load("/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_90_pct/20210430_1745_X2_FAQ15454_23428362/kmers_guppy_4000_dwell/chunks.npy")
idx2kmer = config['idx2kmer']
effective_kmers = get_effective_kmers("!MA",idx2kmer)

figs,axs = plt.subplots(ncols = 3,sharey = True,figsize = (10,5))
sns.distplot(np.mean(chunk_m6A_renorm,axis = 1),ax = axs[0],label = "m6A-Eva")
sns.distplot(np.mean(chunk_control_renorm,axis = 1),ax = axs[0],label = "control-Eva")
axs[0].legend()
sns.distplot(np.mean(chunk_m6A_renorm,axis = 1),ax = axs[1],label = "m6A-Eva")
sns.distplot(np.mean(chunk_control_xron,axis = 1),ax = axs[1],label = "control-NA12878")
axs[1].legend()
sns.distplot(np.mean(chunk_m6A90,axis = 1),ax = axs[2],label = "m6A-NA12878")
sns.distplot(np.mean(chunk_control_xron,axis = 1),ax = axs[2],label = "control-xron-NA12878")
axs[2].legend()
plt.legend()