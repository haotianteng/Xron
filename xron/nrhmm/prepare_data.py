"""
Created on Wed Apr 13 02:41:15 2022

@author: Haotian Teng
"""
import os
import sys
import toml
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from typing import List, Union
from xron.utils.seq_op import fast5_iter,norm_by_noisiest_section,dwell_normalization,combine_normalization

class Extractor(object):
    def __init__(self,k,alphabeta):
        self.k = k
        self.ab_dict = {x:idx for idx,x in enumerate(alphabeta)}
        self.ba_dict = {idx:x for idx,x in enumerate(alphabeta)}
        self.N_BASE = len(alphabeta)
        self.kmer2idx_dict,self.idx2kmer = self.build_kmer_dict()
        
    def kmer_decode(self,
                    sequence:str,
                    signal:np.array,
                    seq_pos:np.array) -> List[str]:
        """
        Generate a kmer sequence given the sequence and the relavant sequence
        position.

        Parameters
        ----------
        sequence : str
            The sequence.
        signal : np.array
            The corresponding reversed signal related to the ref_sig_idx.
        seq_pos : np.array
            A array with same length of the sequence, ith element P_i means the
            ith base in sequence is at P_i location in the signal.

        Returns
        -------
        kmer_seq : List[str]
            Return the list of the kmers.

        """
        kmer_seq = []
        curr_kmer = ''
        kmer_range = seq_pos[(self.k-1)//2:-(self.k//2-1)]
        seq_duration = seq_pos[1:] - seq_pos[:-1]
        seq_duration = self._smooth_zero(seq_duration)
        segemented_signal = signal[kmer_range[0]:kmer_range[-1]]
        for idx in np.arange((self.k-1)//2,len(sequence) - self.k//2):
            start = idx-(self.k-1)//2
            curr_kmer = sequence[start:start+self.k]
            kmer_seq += [self.kmer2idx_dict[curr_kmer]]*seq_duration[idx]
        kmer_seq = np.asarray(kmer_seq)
        return segemented_signal,kmer_seq
    
    def _smooth_zero(self,duration_vector:np.array):
        """
        Smooth out the zero duration to prevent >1 jump in kmer train.

        Parameters
        ----------
        duration_vector : np.array
            The duration vector.

        """
        zero_count = (duration_vector==0).sum()
        pos_idx = np.where(duration_vector > 1)[0]
        compensate_idx = np.random.choice(pos_idx,zero_count,replace = False)
        duration_vector[compensate_idx] -=1
        duration_vector[duration_vector==0 ] +=1
        return duration_vector
    
    def _kmer2idx(self,kmer:Union[List[int],str]):
        "Map the kmer to index, e.g. AAAAA -> 00000, AAAAC -> 00001, AAAAG -> 00002, AAAAT -> 00003, ..."
        if type(kmer) == str:
            kmer = self._num_seq(kmer)
        if len(kmer)<self.k:
            raise ValueError("Except a %dmer, but got %d"%(self.k,len(kmer)))
        multi = 1
        idx = 0
        for base in kmer[::-1]:
            idx += (int(base))*multi
            multi = multi * self.N_BASE
        return idx
    
    def build_kmer_dict(self):
        kmer2idx_dict, idx2kmer = {},[]
        for i in np.arange(self.N_BASE**self.k):
            int_list = []
            for _ in np.arange(self.k):
                int_list.append(i%self.N_BASE)
                i = i//self.N_BASE
            int_list = int_list[::-1]
            kmer = ''.join([self.ba_dict[x] for x in int_list])
            idx = self._kmer2idx(int_list)
            kmer2idx_dict[kmer] = idx
            idx2kmer.append(kmer)
        return kmer2idx_dict, idx2kmer
            
    
    def _num_seq(self,sequence:str):
        return [self.ab_dict[x] for x in sequence]
    
    
def chop(arr,chunk_length,padding = True,pad_values = 0):
    read_len = len(arr)
    chunks = np.split(arr,np.arange(0,read_len,chunk_length))[1:]
    length = [len(x) for x in chunks]
    if padding:
        last_chunk = chunks[-1]
        chunks[-1]= np.pad(last_chunk,(0,chunk_length-len(last_chunk)),'constant',constant_values = (pad_values,pad_values))
    return chunks,length

def extract(args):
    iterator = fast5_iter(args.input,mode = 'r')
    read_count = {"No basecall":0, "Kmer decode failed due to low quality":0,"Succeed":0}
    extractor = Extractor(k = args.k, alphabeta = args.alphabeta)
    config = {"k":args.k,
              "alphabeta":args.alphabeta,
              "chunk_len":args.chunk_len,
              "kmer2idx_dict":extractor.kmer2idx_dict,
              "idx2kmer":extractor.idx2kmer}
    signal_chunks, durations, kmer_chunks, meta_info = [],[],[],[]
    loop_obj = tqdm(iterator)
    for read_h,signal,fast5_f,read_id in loop_obj:
        loop_obj.set_postfix_str("no entry: %d, Kmer decode failed: %d, succeed reads:%d"%(read_count["No basecall"],read_count["Kmer decode failed due to low quality"],read_count["Succeed"]))
        try:
            seg_h = read_h['Analyses/Segmentation_%s'%(args.basecall_entry)]
            ref_sig_idx = np.asarray(seg_h["Reference_corrected/ref_sig_idx"])
            ref_seq = str(np.asarray(seg_h["Reference_corrected/ref_seq"]).astype(str))
            # basecall_entry = args.basecall_entry
        except KeyError:
            if args.alternative_entry:
                try:
                    seg_h = read_h['Analyses/Segmentation_%s'%(args.alternative_entry)]
                    ref_sig_idx = np.asarray(seg_h["Reference_corrected/ref_sig_idx"])
                    ref_seq = str(np.asarray(seg_h["Reference_corrected/ref_seq"]).astype(str))
                    # basecall_entry = args.alternative_entry #The real entry is recorded here for future possible expansion of this module.
                except KeyError:
                    read_count["No basecall"]+=1
                    continue
            else:
                read_count["No basecall"]+=1
                continue
        if args.meth:
            ref_seq.replace("A","M")
        try:
            segmented_signal,kmer_seqs = extractor.kmer_decode(ref_seq, signal[::-1], ref_sig_idx)
        except ValueError:
            read_count["Kmer decode failed due to low quality"]+=1
            continue
        if args.normalization == "dwell":
            norm_signal = dwell_normalization(segmented_signal, kmer_seqs)
        elif args.normalization == "combine":
            norm_signal = combine_normalization(segmented_signal, kmer_seqs)
        elif args.normalization == "noise":
            norm_signal,_,_ = norm_by_noisiest_section(segmented_signal)
        else:
            raise ValueError("Normalization method can only be dwell, combine or noise.")
        curr_chunks,curr_duration = chop(norm_signal,args.chunk_len)
        curr_kmers,_ = chop(kmer_seqs,args.chunk_len,pad_values = -1)
        signal_chunks += curr_chunks
        durations += curr_duration
        kmer_chunks += curr_kmers
        meta_info += [(fast5_f,read_id)]*len(curr_chunks)
        read_count["Succeed"] += 1
        if args.max_n and (args.max_n > 0) and (len(signal_chunks)>args.max_n):
            signal_chunks = signal_chunks[:args.max_n]
            kmer_chunks = kmer_chunks[:args.max_n]
            durations = durations[:args.max_n]
            meta_info = meta_info[:args.max_n]
            break
    np.save(os.path.join(args.output,'chunks.npy'),np.asarray(signal_chunks))
    np.save(os.path.join(args.output,'kmers.npy'),np.asarray(kmer_chunks))
    np.save(os.path.join(args.output,'durations.npy'),np.asarray(durations))
    np.savetxt(os.path.join(args.output,'meta.csv'),meta_info,fmt="%s")
    with open(os.path.join(args.output,'config.toml'),'w+') as f:
        toml.dump(config,f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract kmer information for training the NRHMM.')
    parser.add_argument("-i","--input", required = True, type = str,
                        help = "The input folder contains the fast5 files.")
    parser.add_argument("-o","--output", required = True, type = str,
                        help = "The output folder.")
    parser.add_argument("--chunk_len", default = 4000, type = int,
                        help = "The length of the chunk.")
    parser.add_argument("--basecall_entry", default = "000",type = str,
                        help = "The basecall entry.")
    parser.add_argument("--alternative_entry", default = None, type = str,
                        help = "The alternative entry to try when there is no main entry.")
    parser.add_argument("-k",type = int, default = 5,
                        help = "The k of k-mer.")
    parser.add_argument("--alphabeta", type = str, default = 'ACGTM',
                        help = "Give the order of the necleotides alphabet from 0 to n")
    parser.add_argument("--normalization",type = str, default = "dwell",
                        help = "The normalization method used, can be dwell, combine or noise.")
    parser.add_argument('--max_n',
                        default = None,
                        type=int,
                        help="The maximum number of the segments to be extracted")
    parser.add_argument('--methylated',action = "store_true", dest = "meth",
                        help="This is a fully or high-proportion methylated dataset, so we replace all A with M.")
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.output,exist_ok=True)
    extract(args)

# if __name__ == "__main__":
#     e = Extractor(5, 'ACGT')
#     seq = 'AAACGTCGTGTTT'
#     sig_start = 13
#     segemented_signal,kmer_seq = e.kmer_decode(seq, signal = np.random.rand(200), seq_pos = np.asarray([sig_start+5*x+x**2 for x in np.arange(len(seq))]))
