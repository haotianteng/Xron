# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import toml
rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Arial"
#make font bold
plt.rcParams["font.weight"] = "bold"
BRIDGE_F=os.environ['SCRATCH']
chunks_c = np.load(f"{BRIDGE_F}/ime4_Yearst/IVT/control/rep2/kmers_guppy_4000_noise/chunks.npy",mmap_mode = "r")
paths_c = np.load(f"{BRIDGE_F}/ime4_Yearst/IVT/control/rep2/kmers_guppy_4000_noise/path.npy",mmap_mode = "r")
chunks_m = np.load(f"{BRIDGE_F}/ime4_Yearst/IVT/m6A/rep2/kmers_guppy_4000_noise/chunks.npy",mmap_mode = "r")
paths_m = np.load(f"{BRIDGE_F}/ime4_Yearst/IVT/m6A/rep2/kmers_guppy_4000_noise/path.npy",mmap_mode = "r")
config = toml.load(f"{BRIDGE_F}/ime4_Yearst/IVT/control/rep2/kmers_guppy_4000_noise/config.toml")
idx2kmer = config['idx2kmer']
fig,axs = plt.subplots(nrows = 2, figsize = (20,6.6))
axs[0].axis('off')
axs[1].axis('off')

def plot_segmentation(chunks,paths,test_i,T,ax,control = False,m6A = False):
    path = paths[test_i][:T]
    chunk = chunks[test_i][:T]
    move = path[1:] != path[:-1]
    pos = np.where(move)[0]
    pos = np.append(pos,T)
    ax.plot(chunk,color = "#A0B1BA",lw = 0.7)
    pre_p = 0
    color_map = {"A":"#FFC61E","G":"#AF58BA","C":"#009ADE","T":"#FF1F59","M":"#F28522"}
    i = 0
    for p in pos:
        b = idx2kmer[path[p-1]][2]
        if control:
            if b == "M":
                b = 'A'
        elif m6A:
            if b == "A":
                b = 'M'
        ax.axvspan(pre_p+0.01, p-0.01, color=color_map[b], alpha=1.0,lw = 0)
        offset = 15
        if p-pre_p<offset:
            continue
        else:
            ax.axvline(x = pre_p,color = "black",lw = 1,ls = "--")
            c = "#E9002D" if (b == "M" or b=="A") else "black"
            ax.text(x = pre_p,y =np.min(chunk)-0.58,s = b,color = c )
        pre_p = p
        i+=1
        
test_i = 4
T = 2000
plot_segmentation(chunks_c,paths_c,test_i,T,axs[0],control = True)
plot_segmentation(chunks_m,paths_m,test_i,T,axs[1],m6A = True)
#bold font
axs[0].set_title("unmodified",size = 18,fontweight = "bold")
axs[1].set_title("m6A modified",size = 18,fontweight = "bold")
# axs[0].set_yticklabels([])
# axs[1].set_yticklabels([])
# axs[0].set_xticklabels([])
# axs[1].set_xticklabels([])
# axs[0].set_ylabel("Current")
# axs[1].set_ylabel("Current")

out_f =f"{BRIDGE_F}/Xron_Project"
print(f"Saving to {out_f}")
fig.savefig(os.path.join(out_f,"segmentation.png"),transparent = True,bbox_inches = "tight",pad_inches = 0.1)
fig.savefig(os.path.join(out_f,"segmentation.pdf"),transparent = True,bbox_inches = "tight",pad_inches = 0.1,format = "pdf")
fig.savefig(os.path.join(out_f,"segmentation.eps"),transparent = True,bbox_inches = "tight",pad_inches = 0.1,format = "eps")