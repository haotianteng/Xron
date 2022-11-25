"""
Created on Fri Jun 17 23:00:01 2022

@author: Haotian Teng
"""
import os 
import sys
import toml
import random
import argparse
import numpy as np
from typing import List
import itertools
from tqdm import tqdm
from xron.utils.seq_op import Methylation_DP_Aligner
def kmers2seq(kmers,idx2kmer,return_pos = False):
    merged = [g for g,_ in itertools.groupby(kmers)]
    seqs = [idx2kmer[x][0] for x in merged]
    if return_pos:
        pos = [len(list(G)) for _,G in itertools.groupby(kmers)]
        return ''.join(seqs) + idx2kmer[merged[-1]][1:],pos
    return ''.join(seqs) + idx2kmer[merged[-1]][1:]

def transfer(args):
    kmers = np.load(os.path.join(args.input,"kmers.npy"),mmap_mode = "r")
    durations = np.load(os.path.join(args.input,"durations.npy"),mmap_mode = "r")
    config = toml.load(os.path.join(args.input,"config.toml"))
    seqs,seq_lens = [],[]
    seqs = [kmers2seq(x[:durations[i]],config['idx2kmer']) for i,x in tqdm(enumerate(kmers))]
    seq_lens = [len(x) for x in seqs]
    np.save(os.path.join(args.input,'seqs.npy'),seqs)
    np.save(os.path.join(args.input,'seq_lens.npy'),seq_lens)

class LinkingAlignment(object):
    def __init__(self,aln_seq,duration,aln_ref):
        """Linking the alignment sequence and reference sequence together.
        duration:    d0 d1 0  0  d2 d3 d4 d5
                     |  |  |  |  |  |  |  |
        seq          A->C->_->_->G->A->C->T
                     |  |  |  |  |  |  |  |
        ref          A->_->C->T->_->_->C->T
        """
        self.root = [SeqNode('$',0),RefNode('$')]
        j = 0 #The index of the duration
        #Linking the sequence
        prev = self.root[0]
        for i,c in enumerate(aln_seq):
            curr_d = duration[j] if c !='_' else 0
            if c != '_':
                j+=1
            curr = SeqNode(c,curr_d)
            curr.prev = prev
            prev.next = curr
            prev = curr
        #Linking the reference
        prev = self.root[1]
        for i,c in enumerate(aln_ref):
            curr = RefNode(c)
            curr.prev = prev
            prev.next = curr
            prev = curr
        #Linking the sequence and reference
        curr_q = self.root[0]
        curr_r = self.root[1]
        while curr_q is not None:
            curr_q.ref = curr_r
            curr_r.q = curr_q
            curr_q = curr_q.next
            curr_r = curr_r.next
    
    @property
    def seq(self):
        curr = self.root[0].next
        seq = ''
        while curr is not None:
            seq += curr.c
            curr = curr.next
        return seq
    
    @property
    def ref(self):
        curr = self.root[1].next
        seq = ''
        while curr is not None:
            seq += curr.c
            curr = curr.next
        return seq
    
    @property
    def durations(self):
        curr = self.root[0].next
        durations = []
        while curr is not None:
            durations.append(curr.duration)
            curr = curr.next
        return durations

    def insertion_traverse_fix(self):
        #Fix the insertion in the sequence.
        curr = self.root[0].next
        while curr is not None:
            if curr.ref.c == '_':
                curr = self._carve_up_even(curr)
            else:
                curr = curr.next
    
    def deletion_traverse_fix(self):
        #Fix the deletion in the reference.
        curr = self.root[1].next
        while curr is not None:
            if curr.q.c == '_':
                curr = self._apportion_even(curr.q)
                curr = curr.ref if curr is not None else None
            else:
                curr = curr.next

    def mismatch_traverse_fix(self):
        #Fix the mismatch in the sequence.
        curr = self.root[0].next
        while curr is not None:
            if curr.c != curr.ref.c and curr.c != '_' and curr.ref.c != '_':
                curr.c = curr.ref.c
            curr = curr.next
    
    def _carve_up_left(self,seq_node):
        """Carve up chosen element into its left node, correction for insertion in the sequence.
        """
        if seq_node.prev is None or seq_node.prev.c == '$':
            #There is no prefix node
            seq_node.next.duration += seq_node.duration
            seq_node.prev.next = seq_node.next
            seq_node.next.prev = seq_node.prev
            seq_node.ref.prev.next = seq_node.ref.next
            seq_node.ref.next.prev = seq_node.ref.prev
        else:
            #There is no suffix node
            seq_node.prev.duration += seq_node.duration
            seq_node.prev.next = seq_node.next
            seq_node.next.prev = seq_node.prev
            seq_node.ref.prev.next = seq_node.ref.next
            seq_node.ref.next.prev = seq_node.ref.prev
        del seq_node.ref
        curr_node = seq_node.next
        del seq_node
        return curr_node

    def _carve_up_right(self,seq_node):
        """Carve up chosen element into its left node, correction for insertion in the sequence.
        """
        if seq_node.next is None:
            #There is no suffix node
            seq_node.prev.duration += seq_node.duration
            seq_node.prev.next = seq_node.next
            seq_node.next.prev = seq_node.prev
            seq_node.ref.prev.next = seq_node.ref.next
            seq_node.ref.next.prev = seq_node.ref.prev
        else:
            #carve to suffix node.
            seq_node.next.duration += seq_node.duration
            seq_node.prev.next = seq_node.next
            seq_node.next.prev = seq_node.prev
            seq_node.ref.prev.next = seq_node.ref.next
            seq_node.ref.next.prev = seq_node.ref.prev
        del seq_node.ref
        curr_node = seq_node.next
        del seq_node
        return curr_node

    def _carve_up_even(self,seq_node):
        """Carve up chosen element into its neighbourhood evenly, correction for insertion in the sequence.
        """
        if seq_node.prev is None or seq_node.prev.c == '$':
            #There is no prefix node
            seq_node.next.duration += seq_node.duration
            if seq_node.prev:
                seq_node.prev.next = seq_node.next
                seq_node.ref.prev.next = seq_node.ref.next
            seq_node.next.prev = seq_node.prev
            seq_node.ref.next.prev = seq_node.ref.prev
        elif seq_node.next is None:
            #There is no suffix node
            seq_node.prev.duration += seq_node.duration
            seq_node.prev.next = seq_node.next
            seq_node.ref.prev.next = seq_node.ref.next
        else:
            #Otherwise divide the current node evenly
            seq_node.prev.duration += seq_node.duration//2
            seq_node.next.duration += seq_node.duration//2
            random.choice([seq_node.prev,seq_node.next]).duration += seq_node.duration%2
            seq_node.prev.next = seq_node.next
            seq_node.next.prev = seq_node.prev
            seq_node.ref.prev.next = seq_node.ref.next
            seq_node.ref.next.prev = seq_node.ref.prev
        del seq_node.ref
        curr_node = seq_node.next
        del seq_node
        return curr_node

    def _apportion_even(self,seq_node):
        """Apportion duration from nearby nodes, correction for deletion in the sequence."""
        seq_node.c = seq_node.ref.c
        if seq_node.prev.c != '$':
            #Borrow from the previous node
            share_node = seq_node.prev
            share_ratio = 3
            while share_node is not None and share_node.duration < 2:
                share_node = share_node.prev
                share_ratio += 1
            if share_node is not None:
                share = max(1,int(share_node.duration//share_ratio))
                seq_node.duration += share
                share_node.duration -= share
        else:
            #The insertion is happened at the beginning of the sequence, delete it.
            seq_node.prev.next = seq_node.next
            seq_node.next.prev = seq_node.prev
            seq_node.ref.prev.next = seq_node.ref.next
            seq_node.ref.next.prev = seq_node.ref.prev
            del seq_node.ref
            curr_node = seq_node.next
            del seq_node
            return curr_node

        if seq_node.next is not None:
            #Divide duration from the next node
            share_node = seq_node.next
            share_ratio = 3
            while share_node is not None and share_node.duration < 2:
                share_node = share_node.next
                share_ratio += 1
            if share_node is None:
                #There is no next node to borrow from, skip it
                if seq_node.duration > 0:
                    seq_node.c = seq_node.ref.c
                    return seq_node.next
                else:
                    #The node can't borrow duration from anywhere. delete it.
                    seq_node.prev.next = seq_node.next
                    seq_node.next.prev = seq_node.prev
                    seq_node.ref.prev.next = seq_node.ref.next
                    seq_node.ref.next.prev = seq_node.ref.prev
                    del seq_node.ref
                    curr_node = seq_node.next
                    del seq_node
                    return curr_node
            share = max(1,int(share_node.duration//share_ratio))
            seq_node.duration += share
            share_node.duration -= share
        seq_node.c = seq_node.ref.c
        return seq_node.next

class SeqNode(object):
    def __init__(self,c,duration):
        self.c = c
        self.duration = duration
        self.next = None
        self.prev = None
        self.ref = None #Linking to a RefNode

class RefNode(object):
    def __init__(self,c):
        self.c = c
        self.q = None #Linking to a SeqNode
        self.next = None
        self.prev = None

def fixing_looping_path(path,seq,idx2kmer:List,modified_base,canonical_base):
    """Fixing the looping path like AACCAACCAACCA to AACCA due to circle on HMM.
    """
    kmer_n = len(idx2kmer[0])
    decoded,duration = kmers2seq(path,idx2kmer,return_pos = True)
    if decoded == seq:
        return path,False
    duration += [0]*(kmer_n-1)
    aligner = Methylation_DP_Aligner(base_alternation = {modified_base:canonical_base})
    x,y = aligner.align(decoded, seq)
    aln = LinkingAlignment(x,duration,y)
    aln.insertion_traverse_fix()
    aln.deletion_traverse_fix()
    aln.mismatch_traverse_fix()
    fix_seq,fix_duration = aln.seq,aln.durations
    if len(fix_seq) <= kmer_n:
        return path,None #The path is too short to be fixed
    fix_path = seq2path(fix_seq,fix_duration,idx2kmer)
    return np.asarray(fix_path),True

def seq2path(seq,duration,idx2kmer:List):
    """Convert a sequence to a path.
    """
    kmer_n = len(idx2kmer[0])
    path = []
    for i in range(0,len(seq)-kmer_n):
        kmer = seq[i:i+kmer_n]
        path += [idx2kmer.index(kmer)]*duration[i]
    path += [idx2kmer.index(seq[-kmer_n:])]*(sum(duration[-kmer_n:]))
    return path

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Transfer the kmer-seq to sequences.')
    # parser.add_argument('-i', '--input', required = True,
    #                     help = "The folder contains the kmers.npy and durations.npy files.")
    # args = parser.parse_args(sys.argv[1:])
    # transfer(args)
    config = toml.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_noise/config.toml")
    seqs = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_noise/seqs.npy")
    paths = np.load("/home/heavens/bridge_scratch/ime4_Yearst/IVT/control/rep1/kmers_guppy_4000_noise/path.npy")
    fix_path = fixing_looping_path(paths[11],seqs[11],idx2kmer = config['idx2kmer'],canonical_base = 'A',modified_base = 'M')