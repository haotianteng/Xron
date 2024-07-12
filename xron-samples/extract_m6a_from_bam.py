#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:39:20 2022
This script is used to generate a summary table of m6A sites from tagged SAM file
@author: heavens
"""
import os
import re
import sys
import pysam
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import compress

FLAGS = {"5mC":"C+m","5hmC":"C+h","5fC":"C+f","5caC":"C+c",
         "5hmU":"T+g","5fU":"T+e","5caU":"T+b",
         "m6A":"A+a",
         "8oxoG":"G+o",
         "Xao":"N+n"}
complementary_base = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
complementary_base_rna = {'A':'U','U':'A','C':'G','G':'C','N':'N'}
motif_reg = {"A":"A","C":"C","G":"G","T":"T","U":"U","R":"[AG]","Y":"[CT]","M":"[AC]","K":"[GT]","S":"[GC]","W":"[AT]","B":"[CGT]","D":"[AGT]","H":"[ACT]","V":"[ACG]","N":"[ACGT]"}
def get_reg(motif):
    reg = ""
    for i in motif:
        reg += motif_reg[i]
    return reg

def rc(seq, rna = False):
    if rna:
        return ''.join(complementary_base_rna[x] for x in seq.upper()[::-1])
    else:
        return ''.join(complementary_base[x] for x in seq.upper()[::-1])

class MethParser(object):
    def __init__(self,modified):
        self.modified = modified
        self.flag = FLAGS[modified]

    def parse_MMflag(self,read):
        """
        Parse the MM tag in the SAM file.
        Args:
            read: pysam AlignedSegment object.
        Returns:
            MMs:dict the methylation position for each modification type.
                {modification_code:np.array([position1,position2,...])}
            MLs:dict the methylation likelihood for each modification type.
        """
        try:
            MM_tag = read.get_tag('MM') # The MM tag in the SAM file, start from methylation code, e.g. "A+a?,10,2,3;"
            ML_array = read.get_tag('ML') # A uint8 array of methylation likelihood for each position in MM tag.
        except KeyError:
            return None,None
        MMs = {}
        MLs = {}
        MM_tags = MM_tag.strip(';').split(';') # ['A+a?,10,2,3','C+m?,5,6,7']
        for MM_tag in MM_tags:
            code,position = self._parse_MMtag(MM_tag)
            MMs[code] = np.asarray(position,dtype = int)
        if len(ML_array) != sum(len(x) for x in MMs.values()):
            return None,None
        for key,val in MMs.items():
            c = len(val)
            MLs[key] = ML_array[:c]
            del ML_array[:c]
        return MMs,MLs

    def _parse_MMtag(self,MM_tag):
        #Parse single MM tag for each modification
        MM_tag_split = MM_tag.split(',')
        title = MM_tag_split[0].strip('.').strip('?')
        unmo_base,mo_base = title.split('+')
        if len(mo_base)>1:
            raise ValueError("Multi modification format is currently not supported.")
        if len(MM_tag_split) == 1:
            mo_pos = []
        elif MM_tag_split[1] == '':
            mo_pos = []
        else:
            mo_pos = [int(x) for x in MM_tag_split[1:]]
        return title,mo_pos

    def get_hc(self,read):
        cigar_string = read.cigarstring
        # Find all occurrences of a number followed by 'H' at the start or end
        start_match = re.match(r'^(\d+)H', cigar_string)
        end_match = re.search(r'.(\d+)H$', cigar_string)
        
        start_h = int(start_match.group(1)) if start_match else 0
        end_h = int(end_match.group(1)) if end_match else 0
        
        return start_h, end_h

    def expand_compact_mm(self,mm,ml,pad_val = 0):
        """
        Expand the compact MM probability into a long one.
        e.g. with MM tag A+a?,3,2,3; and ML tag 24,125,33
        The expanded ML tag will be [0,0,0,24,0,0,125,0,0,0,33]

        Parameters
        ----------
        mm : array
            The compact methylation jump array.
        ml : array
            The compact methylation likelihood array.

        Returns
        -------
        expand_ml : array
            The expanded methylation likelihood array.

        """
        expand_ml = []
        for i,jump in enumerate(mm):
            expand_ml+= [pad_val]*jump
            expand_ml+= [ml[i]]
        return np.asarray(expand_ml,dtype = np.uint8)

    def pad_ml(self,ml,read,pad_val = 0):
        """
        Pad the methylation likelihood array to the same length as the read sequence.

        Parameters
        ----------
        ml : array
            The methylation likelihood array.
        read : pysam.libcalignedsegment.AlignedSegment
            The read sequence.

        Returns
        -------
        ml : array
            The padded methylation likelihood array.

        """
        canonical_base = self.flag.split('+')[0]
        seq = read.query_sequence
        if read.is_reverse:
            seq = rc(seq)
        seq = np.asarray(list(seq))
        num_pos = sum(seq == canonical_base)
        if num_pos < 0:
            raise ValueError("The number of canonical base is less than 0.")
        return np.pad(ml,[0,num_pos - len(ml)],mode = 'constant',constant_values = pad_val)
        

    def meth_pileup(self,
                    sam:pysam.libcalignmentfile.AlignmentFile,
                    ref:pysam.libcfaidx.FastaFile,
                    region:str,
                    start:int,
                    end:int,
                    motif = None,
                    span:int = 5,
                    threshold:int = 0):
        """
        Pileup the reads methylation state.

        Parameters
        ----------
        sam : pysam.libcalignmentfile.AlignmentFile
            AlignmentFile generated by pysam.
        ref : pysam.libcfaidx.FastaFile
            Reference index generated by pysam.
        region : str
            The reference region, e.g. chr1.
        start : int
            The start position of the reference to pileup.
        end : int
            The end position of the reference to pileup.
        motif : str, optional
            Only sequence with this motif will be reported. The default is None.
        span : int, optional
            The span of each reference base to record in the pileup. The default is 5.
        threshold : int, optional
            The threshold of the methylation likelihood site to be collected, range from 0 to 255.
            It's a integer coded probability, [0,255]->[0,1]
            The default is 0. Will collect all sites.

        Returns
        -------
        A dictionary contains the pileup information:
            refpos: The reference position of the methylation site.
            refbase: The reference base at the methylation site.
            qbase: The query base at the methylation site.
            read_id: The id of the sequencing read this entry coming from.
            ml: The methylation likelihood of the query base.
            direction: The direction of the read.
            refseq: The reference sequence of the methylation site.

        """
        motif_re = motif_re = re.compile(get_reg(motif)) if motif is not None else None
        b = self.flag.split('+')[0]
        ref_len = ref.get_reference_length(region)
        reads = sam.fetch(region,start,end)
        refpos,qpos,refbase,qbases,direction,refseq,ids = [],[],[],[],[],[],[]
        mms = np.zeros(0,dtype = np.uint8)
        errors = {"Passed":0,"Inconsistent_ML":0,"Corrupted_MM":0}
        with tqdm() as t:
            for read in reads:
                read_str = read.tostring()
                id = read_str.split('\t')[0]

                ###Parsing the MM flag and extract ML likelihood and do sanity check
                mm,ml = self.parse_MMflag(read)
                if mm is None or (self.flag not in mm):
                    errors['Corrupted_MM']+=1
                    continue
                mm,ml = mm[self.flag],ml[self.flag]
                ml = self.expand_compact_mm(mm,ml)
                try:
                    ml = self.pad_ml(ml,read) #pad the ml array to the same number of bases in the query sequence
                except ValueError:
                    errors['Inconsistent_ML']+=1
                    continue

                ###Get base mask for bases to be collected
                pos_pair = read.get_aligned_pairs(with_seq = True, matches_only=False) #[(readpos,refpos,ref_base)]
                query_seq = read.query_sequence.upper()
                curr_seq = [query_seq[x[0]] if x[0] is not None else '*' for x in pos_pair]
                curr_seq = np.asarray(curr_seq)
                if read.is_reverse:
                    curr_b = complementary_base[b]
                    ml = ml[::-1]
                else:
                    curr_b = b
                base_mask = curr_seq == curr_b
                if threshold > 0:
                    #chain mask with ml_mask has the size of sum(base_mask)
                    base_mask[base_mask] = ml>=threshold

                    

                ###Summarize the output
                n_hit = sum(base_mask)
                hc_start,hc_end = self.get_hc(read)
                assert n_hit == len(ml), f"{read.cigarstring}, n_hit:{n_hit}, len(ml):{len(ml)}, ml:{ml}, mm:{mm}, ml_tag:{read.get_tag('ML')}"
                filtered_pos = list(compress(pos_pair,base_mask))
                filtered_seq = list(compress(curr_seq,base_mask))
                qpos += [x[0]+hc_start for x in filtered_pos]
                refpos += [x[1] for x in filtered_pos]
                refbase += [x[2].upper() if x[2] is not None else None for x in filtered_pos]
                refseq += [ref.fetch(region,max(x[1]-span,0),min(x[1]+span+1,ref_len)).upper() if x[1] is not None else None for x in filtered_pos]
                
                qbases += filtered_seq
                ids += [id]*n_hit
                direction += ['+']*n_hit if not read.is_reverse else ['-']*n_hit
                errors["Passed"]+=1
                mms = np.concatenate((mms,ml),axis = 0)
                t.postfix = "Region {REGION} Read_id: {ID}, Passed:{Passed}, ML and sequence have different length:{Inconsistent_ML},Corrupted MM tag:{Corrupted_MM}".format(REGION=region,ID=id,**errors)
                t.update()
        refpos = [x if x is not None else -1 for x in refpos]
        refpos = np.asarray(refpos,dtype = np.int32)
        refbase = np.asarray(refbase)
        refseq = np.asarray(refseq)
        ids = np.asarray(ids)
        direction = np.asarray(direction)
        qbases = np.asarray(qbases)
        if len(refpos) == 0:
            return {"refpos":[],"refbase":[],"refseq":[],"qbase":[],"read_id":[],"ml":[],"direction":[],'refseq':[]}
        mask = (refpos>=start)&(refpos<end)
        if motif_re is not None:
            motif_mask = np.asarray([(re.match(motif_re,x) is not None) if x else False for x in refseq])
            mask = np.logical_and(mask,motif_mask)
        return {"refpos":refpos[mask],
                "qpos":qpos[mask],
                'refbase':refbase[mask],
                'qbase':qbases[mask],
                'read_id':ids[mask],
                'likelihood':mms[mask],
                'direction':direction[mask],
                'refseq':refseq[mask],
                'motif':[motif]*sum(mask)}

def main(args):
    sam_f = args.bam
    ref_f = args.ref
    out = args.out
    motif = args.motif
    threshold = args.threshold
    span = args.span
    # Read in m6A sites from SAM file
    parser = MethParser("m6A")
    if motif is not None:
        motif = motif.upper()
        motif_re = re.compile(get_reg(motif))
    samfile = pysam.AlignmentFile(sam_f, "rb")
    reffile = pysam.FastaFile(ref_f)
    out_f = os.path.join(out,"{MOTIF}_m6A_read_summary{SUFFIX}.tsv".format(MOTIF=motif,SUFFIX="" if args.suffix is None else "_"+args.suffix))
    header_written = False
    if os.path.exists(out_f):
        raise FileExistsError("Output file already exists.")
    for i,region in enumerate(samfile.references):
        pileup = parser.meth_pileup(samfile,
                                    reffile,
                                    region,
                                    start = 0,
                                    end = samfile.get_reference_length(region),
                                    motif = motif,
                                    threshold=threshold,
                                    span = span)
        if len(pileup['refpos']) == 0:
            continue
        pileup = pd.DataFrame(pileup)
        pileup.sort_values(by=['refpos'],inplace=True,ignore_index=True)
        pileup['contig'] = region
        #make contig the first column
        pileup = pileup[['contig']+list(pileup.columns)[:-1]]
        pileup.to_csv(out_f,mode = 'a',header = not header_written,index = False,sep = '\t')
        header_written = True

# Read in putative m6A sites from anti-body capture dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse m6A sites from SAM file')
    parser.add_argument('-b','--bam',help='BAM file contain methylation annotation.',required=True)
    parser.add_argument('-r','--ref',help='Reference file',required=True)
    parser.add_argument('-o','--out',help='Output directory',required=True)
    parser.add_argument('-m','--motif',help='Motif to filter',default=None)
    parser.add_argument('--suffix',help='Suffix for output file',default=None)
    parser.add_argument('--threshold',help='Threshold for methylation likelihood',default=0,type=int)
    parser.add_argument('--span',help='Span of reference base to record',default=5,type=int)
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.out,exist_ok=True)
    main(args)
    

    # ref_f = "/home/heavens/bridge_scratch/ime4_Yearst/Yearst/Yeast_sk1.fasta"
    # motif = "DRACH"
    # sam_f = "/home/heavens/bridge_scratch/ime4_Yearst/Yearst/ko_raw_fast5/xron_crosslink/assess/aln.tagged.sorted.bam"
    # out_f = "/home/heavens/bridge_scratch/Xron_Project/"

