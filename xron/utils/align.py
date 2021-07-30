"""
Created on Sun Feb 21 05:22:32 2021

@author: Haotian Teng
"""
# from bwapy import BwaAligner
# from bwapy.libbwa import Alignment
import sys
if 'bwapy' in sys.modules:
    raise ImportError("Loading bwapy again will cause error, please restart " \
                      "the process.")
import bwapy
import re
import numpy as np
from numpy import ndarray
from typing import List, Optional
from xron.utils.seq_op import raw2seq
from itertools import permutations
from collections import defaultdict
from Bio.Seq import Seq

def reverse_complement(seq):    
    return str(Seq(seq).reverse_complement())

class MetricAligner(bwapy.BwaAligner):
    def __init__(self,
                 reference: str,
                 min_len: int = 7,
                 options:str = None
                 ):
        """
        The wrapper aligner class of bwa aligner, measure the quality for 
        batch of output sequences from a basecall.

        Parameters
        ----------
        reference : str
            File path of a reference fasta/fastq file.
        min_len : int, optional
            The minimum sequence length to be aligned. The default is 7.

        """
        # options = '-x ont2d'
        if not options:
            options = '-A 1 -B 1 -k %d -O 3 -T 0'%(min_len)
        super().__init__(index = reference,options = options)
        self.reference = defaultdict(str)
        with open(reference,'r') as f:
            for line in f:
                if line.startswith(">"):
                    key = line.strip().split(' ')[0][1:]
                else:
                    self.reference[key]+= line.strip()
    
    def permute_align(self,
                      raw: ndarray,
                      alphabet:str = "NACGT",
                      mod_base:dict = {},
                      min_len:int = 5,
                      permute:bool = True
                      )->ndarray:
        """
        Align the input sequence with best permutation.

        Parameters
        ----------
        raw : A [N,L] ndarray gives the raw ctc sequence.
            where N is the number of sequences and L is the length of each
            sequence, where 0 alaways mean the blank symbol.
        permute: List[str] Optional
            A permute list, default is None, will try every possible permutation.
        mod_base: Dict Optional
            A dict mapping the modification base to the original base.
        Returns
        -------
        ndarray
            A arra with same length of the sequences, gives the identity scores.

        """
        
        ab = alphabet[1:]
        seqs = raw2seq(raw)
        len_score = np.asarray([len(seq) for seq in seqs])-min_len
        len_score[len_score<0] = 0
        scores = []
        perms = []
        if permute:
            for perm in permutations(ab):
                perm_seqs = ["".join([perm[x-1] for x in seq]) for seq in seqs]
                identities = self.align(perm_seqs)
                identities = identities * len_score
                scores.append(identities)
                perms.append(perm)
        else:
            seqs = ["".join([ab[x-1] for x in seq]) for seq in seqs]
            identities = self.align(seqs)
            identities = identities * len_score
            scores.append(identities)
            perms.append([x for x in ab])
        return scores, perms
        
        
    def align(self,
              sequences: List[str]
              ) -> ndarray:
        """
        Get the align score given a batch of sequences.

        Parameters
        ----------
        sequences : List[str]
            A list of the sequences.
        Returns
        -------
        ndarray
            A array with same length of the sequences, gives the identity
            scores.

        """
        identities = []
        for seq in sequences:
            hits = self.align_seq(seq)
            identity = self._match_score(hits[0])/len(seq) if len(hits)>0 else 0  #Only use the first hit.
            identities.append(identity)
        return np.array(identities)
    
    def _match_score(self, alignment: Optional[bwapy.libbwa.Alignment]) -> int:
        """
        Get the match score given a cigar string.
    
        Parameters
        ----------
        alignment : Optional[Alignment]
            A alignment instance from the bwapy mapping result.
    
        Returns
        -------
        int
            The nubmer of matched bases.
    
        """
        cigar_s = alignment.cigar
        n_match = self._count_cigar(cigar_s,'M')
        return n_match - alignment.NM

    def ref_seq(self,seq:str):
        """
        Return the reference sequence of the first hit of the alignment

        Parameters
        ----------
        seq : str
            The query sequence.

        Returns
        -------
        ref_seq: Related reference sequence 
        ref_idx: Index reference (index relatively to the query sequence).

        """
        hits = self.align_seq(seq)
        if not hits:
            return [],None,None
        hit = hits[0]
        cigar = hit.cigar
        n_deletion = self._count_cigar(cigar,'D')
        n_match = self._count_cigar(cigar,'M')
        ref_len = n_match + n_deletion 
        ref_seq = self.reference[hit.rname][hit.pos:hit.pos+ref_len]
        ref_idx = np.zeros(len(ref_seq))
        increment = [int(x) for x in re.findall(r'(\d+)',cigar)]
        operation = re.findall(r'[A-Za-z]',cigar)
        curr = 0
        curr_query = 0
        for inc,op in zip(increment,operation):
            if op == "S":
                curr_query += inc
            elif op == "M":
                ref_idx[curr:curr+inc] = np.arange(curr_query,curr_query + inc)
                curr += inc
                curr_query += inc
            elif op == "I":
                curr_query += inc
            elif op == "D":
                ref_idx[curr:curr+inc] = curr
                curr += inc
        if hit.orient == '-':
            ref_seq = reverse_complement(ref_seq)
            ref_idx = len(seq) - ref_idx
            ref_idx = ref_idx[::-1]
        return hits,ref_seq,ref_idx
                

    def _count_cigar(self,cigar_string,c = 'M'):
        return sum([int(x) for x in re.findall(r'(\d+)%s'%(c),cigar_string)])
        
if __name__ == "__main__":
    reference_f = '/home/heavens/twilight_hdd1/m6A_Nanopore/references.fasta'
    test_seqs = ['AAGAUUGUGUUGGUCAGUACUCGGUCGACGGGGUGAUUCCUUCAGCCCAGCCUGAUGAAUGCCGACACCUACCGGCUUUUAAGACGCCCAGGUACCAACGAUCAUCGUCGACCGUAAGU',
                'AUAGAUUGUGUUUGUUAGUCGCUUGGUACGCCGAGCUGAAUGAUUGUGGAUAUCGCGAAGCAGGUUGAGCGCAACCCUGGACGACCGUAUUUCGGCGUCAAUAAGCGUCGUGAAGGCAGGUGAAUAUGAGCAUGGCGUUGGGGCUUUCCGAUAUCAAAGGCGCAGAUGAGCGGUCUGGUAUACCGUGGUUUGUGACUGGCAACGGCUUUAACGUUGGCGAUACGCUGCGUUAUCUGCAGGCAAUUGAGCGGCUGGAGAAAAACUGGCUGUCGAUCCGUCCGAACCGGGCGCGAAAUGCCGAAAGUCGAGCGUCCAGCAGGCGUGGCAUAAUGGAUAAUAAAACUGCCCAGCGCGUCGCGAAGAUGAUGAUGUUAAAGAGAUCCGUUUGGAUGAUUGAAGAACUGCGUGUCAGUAUUCGCCCAACGGUUACGGUGCCUUAUCCUAUCUCCGAUAAGCGUAUUUUACAGGCCAUGGAUCAGAUUACGGCCUAAAGCCAGGCGACGAUGAGCUAAAGAGAUCCAGUACAAGUUCGGAACAAAAAGGAAAUUAUGAAUACAACGCCAUUCCAGCGCUUUAAUGAAACCAUUGCUGGCGGUGAAUACACCCAAUGACAGUGGCUGGCGCAGCGUGAUUUUUCCGGCGAACAACCAUCAUCGGAUUCGCGACCGGCGGGCAAUGCGGUGAUCCCACAUCGACGCGGCCUGUCGCAUCGCCAAGCGCAGGCGUUCCACUGAUUGCGGCGGUAUAAUGGGUCACGCCUUCGACGCCGUUUUUGCGCCGUUGAUUGCCCGGCAUCCCGCGCUACCACACUAUCUGCCAACCGGACGCGCUGAAGCCAUCGAUCUCGCGGAUCGCGAACCAGUCUGGCAUAUUCCGGCUGAGAAAUCUGGCUGGAAGGUCGGUCAACUAUUGCGGCGAAAAUGCCCGUCACCUGCGUCAAUCCGCCAGGCGAAAGAAAACAUUAACGGCUAUCGUUGCGAAGACCCCACCAUGCAGCAGCGCACUAUCGCGUUAUUCCGGCGCGUGACAAACGACGACACCGAUGCGUGCUGGCUGAGUUUUUCCUGGUUUGUUCCAGUACUGCGCCACCUUAACGGCGAUACACGUUUUGCCUGUGUUGAAGAGGGCAUGGGACGGUGGAGCGUUAUCUGUACCGAGAUUGCCGGCGAGCUGCCUGCGUGAUGAAACAGGGUAAGCACCGCGCGGUAAAGAUUUUAUUUCACGUUGGAUAUCCCCACACGCGAUUAUUAGAAACGCAUUGGCGGUAUUAGCAGGCAGAUACGACGCUGCGCAGUGCAAUUAGAACAGAGAGCGUUCGCGAAAAAAUCCCGUCUGCGGGUGAACGGGCAAGUGCGACCACCAUAUCUCAUAAAUUUAAUGAAUAUUUAUAAAAGCAAAAUCGUUGUCGCUCACGGUUUCAUUCAGGACGCGCUAUGGGCGGCAAGUAUUCCGGCCUGCAAAUUGGUAUUUUACUGGUUAAGU',
                'AGACUUGUGUUUGGCGGUCGCUUCUAAACCAGCGUAAAAGA',
                'CAGAUAGAUAGUGUUUGUUAGUCGCUUUGGUGCGAAACUCCACUUAUCUACUUACUACGGUCAAAUUAACGAUCAAGCUUUAAGUACGUCGCCAGAUUGUCUGGUUUAACCCUCAAGUGCGGGACAAUUAUUGGGACGAAUACUACGAGUACGGUCGUGUGGCCAUCUCCACCGUCCCCCGCUGUAAUGCUUUUUUUACUCGUCUUAAGGAUAGGUAUCAUCUGACUAGUACCGGCGUGCAAAAGCUUUUUCCAGGAAAGUAGUCAGGAUGAGGCUGGUGAACUGUUACGAAAGAUGAAGGUAAGACCUCAUCCUACAUUUCACAGGCAAGCCAUUAAAGUGCACUGUCUCGUGGCGAUUGCAACUUCGAUCGGCAUCUGAUCAUUGGCCAGCCCUUAGCCGUGUUUCGGUGUGAAGCAGCGUGGCGGGAAUCGCCAUCAUGAUCCCCGCAUGUAGGGCCCUAUCGGAAUUACACCGUCCUAAUGGUUUUUGAAUUGCGGUGGCGCCAGAAAAGA',
                'AGAUUGUGUUUGUUAGUUGCUUAACAAAAUGGUACAGAGAUGAUAGAAAGGCCUGGAGCCGUCUUCCGGGAAUCCAGAACAGAAUCAUCAACAUUUCCACGAAUCUCAAAGAGCUUAUUGAUUUUGGGCUGAGGCAGCUUGGAAGAAAGCAAUCUCCCAGGCAAAAGCAGCUGUUUUUUUAUCCGUCGAACGGUACUGCACUGCUGUCAUGGGCCGACAGGUUCCAGUAGCUCCUUACCAUGCGAACCGGAUCUACCAGUUUGCCAUCUACUCUGACAUUUUGAAGCUCAGGUGCACCAUUUAUUACACUAACUGUGCAGAUUACCAAUCCAAACUGCUUUCAAUUCGGACACAAAAAAAAGGUUUUACAGGUGCCAGCGAGGAUAAGGCUGGUUUAGUCGAACAAGCCAAUGGAGCAUACACUCUUAUGGACGGCUCCGCCUCCCAAAGGCAAGAAAUGUUGUCUUGCUACUGACAGCGGUUCACCACAACGGCUUGGAGAAUCCUGAACAUAAACGAACAUCAGACGUAUUGUUUAUUUGUGCCACUACAGAAAACCUAGGUCUGCAACUGAAAACAUUUUUGCAGAAUCCCAAUGACCAUUUUAACCAUGUUGUAGAACGAUCGCUUAAAGAAAAGGGUGGAUCUGACGACGUUUUAUUUAGGAAAGAAGCAGAACGAAUUAAGAAAAAUCUGAGUGUGCACAUAGAUGUCUAUAAUGCGCUUAUUCAUUCUGCGAAAUUUGAAAUGUGGGACAGCUGAAAUCAAAUGUCCAGUUAGUUUGUGCACACAGGGUUUUUGUUCCAACCUUGACGGACCGAGGUUAAUGAAAUGACUGUUCGGGAUCUGCCGGAUGAAAUCAAGCAAGAAUGGAUGUCCAGCAGCAAAAUACUGCAAAGGAGCAAAGCCAUUUGAAUGUCAUAAUUUUUUCGAUCAUCUCCGAUUAAGCGGAAGAUGAAACAACCGAAUAUUGACGAGGACCCUUUUCAUUUAACUUGUAUCAUAUUGCGAUCGAAGAAAAGGAG',
                'AGAUUGUGUUUGUUAGUUGCUUAACAAAAUGGUACAGAGAUGAUAGAAAGGCCUGGAGCCGUCUUCCGGGAAUCCAGAACAGAAUCAUCAACAUUUCCACGAAUCUCAAAGAGCUUAUUGAUUUUGGGCUGAGGCAGCUUGGAAGAAAGCAAUCUCCCAGGCAAAAGCAGCUGUUUUUUUAUCCGUCGAACGGUACUGCACUGCUGUCAUGGGCCGACAGGUUCCAGUAGCUCCUUACCAUGCGAACCGGAUCUACCAGUUUGCCAUCUACUCUGACAUUUUGAAGCUCAGGUGCACCAUUUAUUACACUAACUGUGCAGAUUACCAAUCCAAACUGCUUUCAAUUCGGACACAAAAAAAAGGUUUUACAGGUGCCAGCGAGGAUAAGGCUGGUUUAGUCGAACAAGCCAAUGGAGCAUACACUCUUAUGGACGGCUCCGCCUCCCAAAGGCAAGAAAUGUUGUCUUGCUACUGACAGCGGUUCACCACAACGGCUUGGAGAAUCCUGAACAUAAACGAACAUCAGACGUAUUGUUUAUUUGUGCCACUACAGAAAACCUAGGUCUGCAACUGAAAACAUUUUUGCAGAAUCCCAAUGACCAUUUUAACCAUGUUGUAGAACGAUCGCUUAAAGAAAAGGGUGGAUCUGACGACGUUUUAUUUAGGAAAGAAGCAGAACGAAUUAAGAAAAAUCUGAGUGUGCACAUAGAUGUCUAUAAUGCGCUUAUUCAUUCUGCGAAAUUUGAAAUGUGGGACAGCUGAAAUCAAAUGUCCAGUUAGUUUGUGCACACAGGGUUUUUGUUCCAACCUUGACGGACCGAGGUUAAUGAAAUGACUGUUCGGGAUCUGCCGGAUGAAAUCAAGCAAGAAUGGAUGUCCAGCAGCAAAAUACUGCAAAGGAGCAAAGCCAUUUGAAUGUCAUAAUUUUUUCGAUCAUCUCCGAUUAAGCGGAAGAUGAAACAACCGAAUAUUGACGAGGACCCUUUUCAUUUAACUUGUAUCAUAUUGCGAUCGAAGAAAAGGAG',
                'AAAGTAAGCCTGCTCGCTAAGTAGTCTTGCCCGCTCGGTGCGCAGTTCAATCGCTTTCAACAG']
    
    aln = MetricAligner(reference_f,options = '-x ont2d -A 1 -B 0')
    scores = aln.align(test_seqs)
    hits = aln.align_seq(test_seqs[1]) #The position is the postiive start position in reference genome even for reverse complement alignment.
    print(scores)
    print(hits)
    aln.ref_seq(test_seqs[1])
    rev_seq = reverse_complement(aln.reference['bsubtilis1'][:100]+'ATACACACA')
    aln.ref_seq(rev_seq)
    