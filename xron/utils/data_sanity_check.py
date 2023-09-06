#%%
import os
import numpy as np
import seaborn as sns
from xron.xron_train_supervised import main
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

#%% Load data
DF = filedialog.askdirectory()
# DF="/home/heavens/bridge_scratch/ELIGOS_dataset/IVT/control/kmers_guppy_4000_noise/"
chunks = np.load(os.path.join(DF,"chunks.npy"),mmap_mode = "r")
durations = np.load(os.path.join(DF,"durations.npy"),mmap_mode = "r")
kmers = np.load(os.path.join(DF,"kmers.npy"),mmap_mode = "r")
seqs = np.load(os.path.join(DF,"seqs.npy"),mmap_mode = "r")
seq_lens = np.load(os.path.join(DF,"seq_lens.npy"),mmap_mode = "r")

# %% Sanity check
sns.distplot(seq_lens)