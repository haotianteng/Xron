"""
Created on Mon Aug 30 15:27:21 2021

@author: Haotian Teng
"""
import os
import numpy as np
from matplotlib import pyplot as plt

def watch_errors(fs):
    errors = []
    for f in fs:
        error_file = os.path.join(f,"errors.csv")
        e = []
        with open(error_file,'r') as f:
            for line in f:
                error = np.float(line.strip())
                if not np.isnan(error):
                    e.append(error)
        errors.append(e)
    return errors

def std_plot(errors:list,axs,**plot_args):
    ls = [len(x) for x in errors]
    max_len = max(ls)
    errors = np.asarray([np.pad(x,(0,max_len-y),'constant',constant_values = np.nan) for x,y in zip(errors,ls)])
    e_mean = np.nanmean(errors,axis = 0)
    e_std =  np.nanstd(errors,axis = 0)
    x = np.arange(len(e_mean))
    axs.plot(x,e_mean,c = plot_args['color'])
    axs.fill_between(x,e_mean - e_std,e_mean + e_std,**plot_args)
    return axs

if __name__ == "__main__":
    prefix = "/home/heavens/bridge_scratch/Xron_models_control/xron_model_supervised_control_dataset_%d_%s_16G"
    adam_fs = [prefix%(x,'Adam') for x in np.arange(4)]    
    sgd_fs = [prefix%(x,'SGD') for x in np.arange(4)]    
    momentum_fs = [prefix%(x,'Momentum') for x in np.arange(4,8,1)]
    adagrad_fs = [prefix%(x,'Adagrad') for x in np.arange(4,8,1)]    
    colors = ['r','g','b','yellow']
    opts = ['Adam','SGD','Momentum','Adagrad']
    axs = plt.subplot()
    for f,c,opt in zip([adam_fs,sgd_fs,momentum_fs,adagrad_fs],colors,opts):
        errors = watch_errors(f)
        std_plot(errors,axs,color = c,label = opt, alpha = .2)
    plt.legend()
    axs.set_xlabel("Training step")
    axs.set_ylabel("Editdistance/Sequence Length")