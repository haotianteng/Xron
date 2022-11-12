"""
Created on Fri May 20 14:53:39 2022

@author: Haotian Teng
"""

import os
import sys
import toml
import torch
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from typing import Dict,List
from xron.utils.seq_op import Methylation_DP_Aligner
from xron.nrhmm.hmm import GaussianEmissions, RHMM
from xron.nrhmm.hmm_input import Kmer2Transition, Kmer_Dataset, Normalizer
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from xron.xron_train_base import DeviceDataLoader

def kmers2seq(kmers,idx2kmer):
    merged = [g for g, _ in itertools.groupby(kmers)]
    seqs = [idx2kmer[x][0] for x in merged]
    return ''.join(seqs) + idx2kmer[merged[-1]][1:]

def print_pore_model(emission_means,idx2kmer,f = None):
    for i,mean in enumerate(emission_means):
        if f:
            f.write("%s: %.2f\n"%(idx2kmer[i],mean))
        else:
            print("%s: %.2f"%(idx2kmer[i],mean))

def get_effective_kmers(selection:str,idx2kmer:List):
    demand_char = [x for x in selection.split("!")[0]]
    filter_char = [x for x in selection.split("!")[1]]
    selected_kmers = [i for i,x in enumerate(idx2kmer) if all([d in x for d in demand_char] + [f not in x for f in filter_char])]
    return selected_kmers

def load(self,save_folder,update_global_step = True):
    self.save_folder = save_folder
    ckpt_file = os.path.join(save_folder,'checkpoint')
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
        if update_global_step:
            self.global_step = int(latest_ckpt.split('-')[1])
    ckpt = torch.load(os.path.join(save_folder,latest_ckpt),
                      map_location=self.device)
    for key,net in ckpt.items():
        if key in self.nets.keys():
            self.nets[key].load_state_dict(net,strict = False)
            self.nets[key].to(self.device)
        else:
            print("%s net is defined in the checkpoint but is not imported because it's not defined in the model."%(key))

def main(args):
    print("Loading data...")
    config = toml.load(os.path.join(args.input,"config.toml"))
    chunks = np.load(os.path.join(args.input,"chunks.npy"),mmap_mode=args.mmap_mode)
    n_samples, sig_len = chunks.shape
    durations = np.load(os.path.join(args.input,"durations.npy"),mmap_mode = args.mmap_mode)
    idx2kmer = config['idx2kmer']
    kmers = np.load(os.path.join(args.input,"kmers.npy"),mmap_mode = args.mmap_mode)
    k2t = Kmer2Transition(alphabeta = config['alphabeta'],
                          k = config['k'],
                          T_max = config['chunk_len'],
                          kmer2idx = config['kmer2idx_dict'],
                          idx2kmer = config['idx2kmer'],
                          neighbour_kmer = 4 ,
                          base_prior = {x:args.transition_prior for x in config['alphabeta']},
                          base_alternation = {"A":"M"}, 
                          kmer_replacement = args.kmer_replacement,
                          out_format = args.transition_type)
    dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
    loader = DataLoader(dataset,batch_size = args.batch_size, shuffle = False)
    loader = DeviceDataLoader(loader,device = args.device)
    
    print("Load the model.")
    ckpt_file = os.path.join(args.model,'checkpoint')
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
    model = torch.load(os.path.join(args.model,latest_ckpt),
                       map_location = torch.device(args.device))
    emission = GaussianEmissions(model['hmm']['emission.means'].cpu().numpy(), model['hmm']['emission.cov'].cpu().numpy())
    hmm = RHMM(emission,
               normalize_transition=False,
               device = args.device,
               transition_operation = args.transition_operation,
               index_mapping = k2t.idx_map if args.transition_operation == "compact" else None)
    
    print("Readout the pore model.")
    with open(os.path.join(args.model,"pore_model"), "w+") as f:
        print_pore_model(model['hmm']['emission.means'],config['idx2kmer'],f)

    aligner = Methylation_DP_Aligner(base_alternation = {'M':'A'})
    if args.effective == "!":
        effective_kmers = None
    else:
        effective_kmers = get_effective_kmers(args.effective, idx2kmer)
    norm = Normalizer(use_dwell = True,
                      effective_kmers=effective_kmers)
    sigs,seqs,paths = [],[],[]
    for i_batch, batch in tqdm(enumerate(loader)):
        renorm_batch = []
        signal_batch = batch['signal']
        duration_batch = batch['duration']
        transition_batch = batch['labels']
        kmers_batch = batch['kmers']
        with torch.no_grad():
            path,logit = hmm.viterbi_decode(signal_batch, duration_batch, transition_batch)
        paths.append(path.cpu().numpy())
        renorm_batch = signal_batch
        for j in tqdm(np.arange(args.renorm+1),desc = "Renorm the signal:"):
            with torch.no_grad():
                path,logit = hmm.viterbi_decode(renorm_batch, duration_batch, transition_batch)
            rc_signal = np.asarray([[hmm.emission.means[x].item() for x in p] for p in path.cpu().numpy()])
            renorm_batch = norm(renorm_batch.cpu().numpy()[:,:,0],rc_signal, duration_batch.cpu().numpy(),path.cpu().numpy())
            renorm_batch = torch.from_numpy(renorm_batch).unsqueeze(dim = -1).to(args.device)
        renorm_batch = renorm_batch.squeeze(dim = -1).cpu().numpy()
        sigs.append(renorm_batch)
        for i in np.arange(args.batch_size):
            if i >= signal_batch.shape[0]:
                continue
            deco_seq = kmers2seq(path.cpu().numpy()[i][:duration_batch[i]],idx2kmer)
            orig_seq = kmers2seq(kmers_batch.cpu().numpy()[i][:duration_batch[i]],idx2kmer)
            deco_aln,orig_aln = aligner.align(deco_seq,orig_seq)
            final_seq = aligner.merge(deco_aln,orig_aln)
            seqs.append(final_seq)
        if args.max_n:
            if len(seqs) >= args.max_n:
                break
    seq_lens = [len(i) for i in seqs]
    sigs = np.vstack(sigs)
    seqs = np.array(seqs)
    seq_lens = np.array(seq_lens)
    paths = np.concatenate(paths,axis = 0)
    np.save(os.path.join(args.input,'chunks_renorm.npy'),sigs[:args.max_n])
    np.save(os.path.join(args.input,'seqs_re.npy'),seqs[:args.max_n])
    np.save(os.path.join(args.input,'seq_re_lens.npy'),seq_lens[:args.max_n])
    np.save(os.path.join(args.input,'path'),paths[:args.max_n])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training RHMM model')
    parser.add_argument("-i","--input", type = str, required = True,
                        help = "Data folder contains the chunk, kmer sequence.")
    parser.add_argument('-m', '--model', required = True,
                        help = "The rhmm model folder.")
    parser.add_argument("-b","--batch_size", type = int, default = 20,
                        help = "The batch size to train.")
    parser.add_argument("--renorm", type = int, default = 1,
                        help = "Number of time to renormalize the signal.")
    parser.add_argument("--device",type = str, default = None,
                        help = "The device used to train the model.")
    parser.add_argument("--max_n",type = int, default = None,
                        help = "The maximum number of reads.")
    parser.add_argument("--certain_methylation",action = "store_false", 
                        dest = "kmer_replacement", 
                        help = "If we are sure about the methylation state.")
    parser.add_argument("--mmap_mode",type = str, default = None,
                        help = "mmap mode when loding numpy data, default is\
                        None which does not enable mmapmode, can be r.")
    parser.add_argument("-e","--effective",type = str, default = "!",
                        help = "A magic string gives the kmers that take into \
                            account when doing normalization, for example, \
                            A!M means kmers that must have A and must not \
                            have M is taken into account.")
    parser.add_argument("--transition_prior",type = float, default = 0.3,
                        help = "The prior probability of transition matrix.")
    parser.add_argument("--transition_operation",type = str, default = "sparse")
    args = parser.parse_args(sys.argv[1:])
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.transition_operation == "compact":
        args.transition_type = "compact"
    else:
        args.transition_type = "sparse"
    main(args)



