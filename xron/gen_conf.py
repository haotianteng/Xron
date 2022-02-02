"""
Created on Sun Dec 26 19:38:25 2021

@author: Haotian Teng
"""
import os
import sys
import argparse
from xron_model import MM_CONFIG

def main(args):
    base_config = MM_CONFIG()
    grid_key = [base_config.CNN['Layers'][0]['out_channels'],
                base_config.CNN['Layers'][1]['out_channels'],
                base_config.CNN['Layers'][2]['out_channels'],
                base_config.RNN['hidden_size'],
                base_config.PORE_MODEL['K']]
    grid = [[4,16,64],
            [16,64,128],
            [256,512,768],
            [256,512,768],
            [2,3,4,5]]
    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='Calling training module.')
#     parser.add_argument('--module', required = True,
#                         help = "The training module to call, can be Embedding, Supervised and Reinforce")
#     parser.add_argument('-o', '--model_folder', required = True,
#                         help = "The folder to save folder at.")
#     args = parser.parse_args(sys.argv[1:])
#     os.makedirs(args.model_folder,exist_ok=True)
#     main(args)