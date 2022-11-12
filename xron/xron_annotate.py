"""
Created on Wed Jul 20 11:04:29 2022

@author: Haotian Teng
"""
import numpy as np
from xron.utils.fastIO import read_fast5
from xron.utils.seq_op import fast5_iter
from itertools import groupby
from tqdm import tqdm
import argparse
import sys

def softmax(logits,axis = -1):
    return np.exp(logits)/np.sum(np.exp(logits),axis = axis,keepdims = True)

def get_posterior(logits,seq,move,canonical_base = 'A',modified_base = 'M',base_list = ['A','C','G','T','M'],n_largest_p = 3):
    """Get the posterior probability of the canonical base and modified base"""
    pos = np.cumsum(move)-1
    posterior = []
    for k,g in groupby(zip(logits,pos),lambda x:x[1]):
        g = np.asarray([x[0] for x in g])
        if k >= len(seq):
            print("Warning, found sequence that is too short, probabily the result is from a overlay>0 basecall.")
            return None
        if seq[k] == canonical_base or seq[k] == modified_base:
            g = softmax(g)
            p_canonical = g[:,base_list.index(canonical_base)+1]
            p_canonical.sort()
            p_canonical = p_canonical[-n_largest_p:].sum()
            p_modified = g[:,base_list.index(modified_base)+1]
            p_modified.sort()
            p_modified = p_modified[-n_largest_p:].sum()
            posterior.append(p_modified/(p_canonical+p_modified))
    return np.asarray(posterior)

def write_modified_probability(args):
    fail_count = 0
    with tqdm() as t:
        for read_h,signal,abs_path,read_id in fast5_iter(args.fast5,mode = 'a'):
            t.postfix = "File: %s, Read: %s, failed: %d"%(abs_path,read_id,fail_count)
            try:
                logits,move,seq = read_fast5(read_h,index = args.basecall_entry)
            except:
                fail_count += 1
                continue
            mod_p = get_posterior(logits,
                                  seq,
                                  move,
                                  canonical_base = args.canonical_base,
                                  modified_base = args.modified_base,
                                  base_list = [x for x in args.alphabeta],
                                  n_largest_p = args.n_largest_p)
            if mod_p is None:
                continue
            result_h = read_h['Analyses/Basecall_1D_%s/BaseCalled_template'%(args.basecall_entry)]
            if 'ModifiedProbability' in result_h:
                del result_h['ModifiedProbability']
            result_h.create_dataset('ModifiedProbability',data = mod_p,dtype = "f")
            result_h['ModifiedProbability'].attrs['alphabet'] = args.alphabeta
            result_h['ModifiedProbability'].attrs['canonical_base'] = args.canonical_base
            result_h['ModifiedProbability'].attrs['modified_base'] = args.modified_base
            result_h['ModifiedProbability'].attrs['n_largest_p'] = args.n_largest_p
            t.update()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--fast5',type = str,required = True,help = 'Path to the fast5 folder')
    parser.add_argument('--basecall_entry',type = str,default = '000',help = 'Basecall entry')
    parser.add_argument('--alphabeta',type = str,default = 'ACGTM',help = 'Alphabet of the basecall model. Default is ACGTM')
    parser.add_argument('--canonical_base',type = str,default = 'A',help = 'Canonical base. Default is A')
    parser.add_argument('--modified_base',type = str,default = 'M',help = 'Modified base. Default is M')
    parser.add_argument('--n_largest_p',type = int,default = 3,help = 'Number of largest probability to use. Default is 3')
    args = parser.parse_args(sys.argv[1:])
    write_modified_probability(args)