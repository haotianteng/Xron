"""
Created on Mon Aug 30 15:27:21 2021

@author: Haotian Teng
"""
import os
from typing import List
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

def watch_errors(fs):
    errors = []
    for f in fs:
        error_file = os.path.join(f,"errors.csv")
        e = []
        with open(error_file,'r') as f:
            for line in f:
                try:
                    error = np.float(line.strip())
                except:
                    print(line.strip())
                if not np.isnan(error):
                    e.append(error)
        errors.append(e)
    return errors

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def std_plot(errors:list,axs,**plot_args):
    ls = [len(x) for x in errors]
    max_len = max(ls)
    errors = np.asarray([np.pad(x,(0,max_len-y),'constant',constant_values = np.nan) for x,y in zip(errors,ls)])
    e_mean = np.nanmean(errors,axis = 0)
    e_std =  np.nanstd(errors,axis = 0)
    x = np.arange(len(e_mean))
    e_mean_smoothed = smooth(e_mean,plot_args['smooth'])
    axs.plot(x,e_mean_smoothed,c = plot_args['color'],linewidth = 1)
    axs.fill_between(x,e_mean - e_std,e_mean + e_std,facecolor = plot_args['color'],alpha = plot_args['alpha'],label = plot_args['label'])
    return axs

if __name__ == "__main__":
    ### Plot multiple optimizers training result
    # prefix = "/home/heavens/bridge_scratch/Xron_models_control/xron_model_supervised_control_dataset_%d_%s_16G"
    # adam_fs = [prefix%(x,'Adam') for x in np.arange(4)]    
    # sgd_fs = [prefix%(x,'SGD') for x in np.arange(4)]    
    # momentum_fs = [prefix%(x,'Momentum') for x in np.arange(4,8,1)]
    # adagrad_fs = [prefix%(x,'Adagrad') for x in np.arange(4,8,1)]    
    # colors = ['r','g','b','yellow']
    # opts = ['Adam','SGD','Momentum','Adagrad']
    # fs = [adam_fs,sgd_fs,momentum_fs,adagrad_fs]
    
    # Plot single training error
    # opts = ['2000-LN-Adam','2000-LN-Adagrad','4000-LN-Adam_1,3','4000-LN-Adam_0,2','4000-LN-Adagrad','2000-BN-Adam','2000-BN-Adagrad']
    # opts = [str(x) for x in range(8)]
    # opts = [str(x) for x in range(4)]
    colors = sns.color_palette()
    
    ##Control plotting
    
    # fs = [['/home/heavens/bridge_scratch/Xron_models_merge_d/xron_model_supervised_merge_Adagrad_transfer_learning_fixD'],
    #       ['/home/heavens/bridge_scratch/Xron_models_merge_d/xron_model_supervised_merge_Adagrad_transfer_learning_deviationCorrected'],
    #        ['/home/heavens/bridge_scratch/Xron_models_DirectTrainOnMerge/xron_model_supervised_control_dataset_%d_Adagrad_16G'%(x) for x in np.arange(8)]]
    # fs = [['/home/heavens/bridge_scratch/Xron_models_merge_d/xron_model_supervised_merge_Adagrad_transfer_learning_deviationCorrected']]
    
    # ### Plot all
    # title = "Control model accuracy"
    # repeat = 4
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_2000L/xron_model_supervised_control_dataset_%d_Adam_16G'%(x) for x in np.arange(repeat)],
    #       ['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_2000L/xron_model_supervised_control_dataset_%d_Adagrad_16G'%(x+4) for x in np.arange(repeat)],
    #       ['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_1_Adam_16G',
    #         '/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_3_Adam_16G'],
    #       ['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_0_Adam_16G',
    #         '/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_2_Adam_16G'],
    #       ['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_%d_Adagrad_16G'%(x+4) for x in np.arange(repeat)],
    #       ['/home/heavens/bridge_scratch/Xron_models_NAIVT/xron_model_supervised_control_dataset_%d_Adam_16G'%(x) for x in np.arange(repeat)],
    #       ['/home/heavens/bridge_scratch/Xron_models_NAIVT/xron_model_supervised_control_dataset_%d_Adagrad_16G'%(x+4) for x in np.arange(repeat)]]
    # opts = ['2000L-Adam'] + ['2000L-Adagrad'] + ['4000L-Adam'] + ['4000L-Adagrad'] + ['Adam'] + ['Adagrad']
    
    # ## Plot 8000L
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_8000L/xron_model_supervised_control_dataset_%d_16G'%(x)] for x in np.arange(8)]
    # opts = np.arange(8)
    # title = "8000L training accuracy"
    
    # ## Plot 4000L
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_%d_Adam_16G'%(x)] for x in np.arange(4)]
    # fs += [['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_%d_Adagrad_16G'%(x+4)] for x in np.arange(4)]
    # opts = ['Adam']*4 + ['Adagrad']*4
    
    # title = "4000L control dataset training curve"
    
    ## Plot 
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_2000L/xron_model_supervised_control_dataset_%d_Adam_16G'%(x)] for x in np.arange(4)]
    # fs += [['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_2000L/xron_model_supervised_control_dataset_%d_Adagrad_16G'%(x+4)] for x in np.arange(4)]
    # opts = ['Adam']*4 + ['Adagrad']*4
    # title = "2000L control dataset training curve"
    
    ## Plot attention model
    repeat = 4
    fs =[['/home/heavens/bridge_scratch/Xron_models_attention_NAIVT/'],
         ['/home/heavens/bridge_scratch/Xron_models_NAIVT_LayerNorm_4000L/xron_model_supervised_control_dataset_1_Adam_16G']]
    opts = ['Attention'] + ['Adam']
    title = "Attention model"
    
    ### Retrain on 100 Methylation
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT+100METH_LayerNorm_4000L/xron_model_%d_CM%d'%(x,x%4+1)] for x in np.arange(8)]
    # opts = ['Control-Methylation-Ratio %d:1, Adam'%(x+1) for x in np.arange(4)] + ['Control-Methylation-Ratio %d:1, AdamW'%(x+1) for x in np.arange(4)]
    # title = "Retrain only the last two layers"
    ###
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT+100METH_LayerNorm_4000L_retrain_all/xron_model_%d_CM%d'%(x,x%4+1)] for x in np.arange(8)]
    # opts = ['Control-Methylation-Ratio %d:1, Adam'%(x+1) for x in np.arange(4)] + ['Control-Methylation-Ratio %d:1, AdamW'%(x+1) for x in np.arange(4)]
    # title = "Retrain the whole NN."

    ###
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT+100METH+90METH_LayerNorm_4000L/xron_model_%d_CM%d'%(x,x//2%2)] for x in np.arange(8)]
    # lrs = ['1e-4','4e-5']
    # opts = ['Control-Methylation-Ratio %d:1, LR:%s'%(x//2%2,lrs[x%2]) for x in np.arange(8)]
    # title = "Retrain only the last two layers"
    ###
    # fs = [['/home/heavens/bridge_scratch/Xron_models_NAIVT+100METH+90METH_LayerNorm_4000L_retrain_all/xron_model_%d_CM%d'%(x,x//2%2)] for x in np.arange(8)]
    # lrs = ['1e-4','4e-5']
    # opts = ['Control-Methylation-Ratio %d:1, LR:%s'%(x//2%2,lrs[x%2]) for x in np.arange(8)]
    # title = "Retrain the whole NN."

    
    
    fig,axs = plt.subplots(figsize = (40,20))
    FONTSIZE = 30
    for f,c,opt in zip(fs,colors,opts):
        errors = watch_errors(f)
        std_plot(errors,axs,color = c,label = opt, alpha = .2, smooth = 0.7)
    plt.legend()
    axs.set_xlabel("Training step",fontsize = FONTSIZE)
    axs.set_ylabel("Validate Error (Editdistance/Sequence Length)",fontsize = FONTSIZE)
    plt.legend(fontsize = FONTSIZE)
    axs.set_ylim([0,1.0])
    plt.xticks(fontsize = FONTSIZE)
    plt.yticks(fontsize = FONTSIZE)
    plt.title(title,fontsize = FONTSIZE)
    prefix = "/home/heavens/bridge_scratch/Xron_models_merge_d/xron_model_supervised_control_dataset_Adagrad_16G"