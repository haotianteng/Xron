"""
Created on Sun Dec 26 19:38:25 2021

@author: Haotian Teng
"""
import os
import sys
import torch
import partial
import argparse
from xron.xron_model import CONFIG,DECODER_CONFIG,CRITIC_CONFIG,MM_CONFIG
from xron.xron_train_supervised import main as supervised_train
from xron.xron_train_variational import main as reinforce_train
from xron.xron_train_embedding import main as embedding_train

optimizers = {'Adam':torch.optim.Adam,
              'AdamW':torch.optim.AdamW,
              'SGD':torch.optim.SGD,
              'RMSprop':torch.optim.RMSprop,
              'Adagrad':torch.optim.Adagrad,
              'Momentum':partial(torch.optim.SGD,momentum = 0.9)}

def main(args):
    class CTC_CONFIG(MM_CONFIG):
        CTC = {"beam_size":5,
               "beam_cut_threshold":0.05,
               "alphabeta": "ACGTM",
               "mode":"rna"}
        
    class TRAIN_EMBEDDING_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "epsilon":0.1,
                 "epsilon_decay":0,
                 "alpha":0.01, #Entropy loss scale factor
                 "keep_record":5,
                 "decay":args.decay,
                 "diff_signal":args.diff}
    
    class TRAIN_SUPERVISED_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "keep_record":5,
                 "eval_size":10000,
                 "optimizer":optimizers[args.optimizer]}
    
    class TRAIN_REINFORCE_CONFIG(CTC_CONFIG):
        TRAIN = {"inital_learning_rate":args.lr,
                 "batch_size":args.batch_size,
                 "grad_norm":2,
                 "epsilon":0.1,
                 "epsilon_decay":0,
                 "alpha":0.01, #Entropy loss scale factor
                 "beta": 1., #Reconstruction loss scale factor
                 "gamma":0, #Alignment loss scale factor
                 "preheat":5000,
                 "keep_record":5,
                 "decay":args.decay,
                 "diff_signal":args.diff}
    train_config = {"Embedding":TRAIN_EMBEDDING_CONFIG,
                    "Supervised":TRAIN_SUPERVISED_CONFIG,
                    "Reinforce":TRAIN_REINFORCE_CONFIG}
    train_module = {"Embedding":embedding_train,
                    "Supervised":supervised_train,
                    "Reinforce":reinforce_train}
    args.config = train_config[args.module]
    train_module[args.module](args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calling training module.')
    parser.add_argument('--module', required = True,
                        help = "The training module to call, can be Embedding, Supervised and Reinforce")
    parser.add_argument('-i', '--chunks', required = True,
                        help = "The .npy file contain chunks.")
    parser.add_argument('-o', '--model_folder', required = True,
                        help = "The folder to save folder at.")
    parser.add_argument('--seq', required = True,
                        help="The .npy file contain the sequence.")
    parser.add_argument('--seq_len', required = True,
                        help="The .npy file contain the sueqnece length.")
    parser.add_argument('--device', default = 'cuda',
                        help="The device used for training, can be cpu or cuda.")
    parser.add_argument('--lr', default = 4e-3, type = float,
                        help="Initial learning rate.")
    parser.add_argument('--batch_size', default = 200, type = int,
                        help="Training batch size.")
    parser.add_argument('--epoches', default = 10, type = int,
                        help = "The number of epoches to train.")
    parser.add_argument('--report', default = 20, type = int,
                        help = "The interval of training rounds to report.")
    parser.add_argument('--load', dest='retrain', action='store_true',
                        help='Load existed model.')
    parser.add_argument('--config', default = None,
                        help = "Training configuration.")
    parser.add_argument('--optimizer', default = "RMSprop",
                        help = "Optimizer to use, can be Adam, AdamW, SGD and RMSprop,\
                            default is RMSprop")
    parser.add_argument('--threads', type = int, default = None,
                        help = "Number of threads used by Pytorch")
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.model_folder,exist_ok=True)
    main(args)