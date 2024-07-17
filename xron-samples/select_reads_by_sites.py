#%%
"""This script is used to gathering reads from a given sites file with a sam index file."""
import os
import sys
import h5py
import pysam
import numpy as np
import pandas as pd
from tqdm import tqdm 
import random
import time
from typing import List
from multiprocessing import Pool,current_process,RLock
from multiprocessing.managers import BaseManager
import sys
from functools import partial
class MultiProcessBars(object):
    def __init__(self,bar_string,l = 40):
        """Maintain multiple progress bars of chiron running
        Args:
            bar_string([string]): List of the names of the bars.
            progress([int/float]): 
        """
        if isinstance(bar_string,str):
            bar_string = [bar_string]
        elif not isinstance(bar_string,list):
            raise ValueError("Bar string must be either string type or list type")
        self.bar_string = bar_string
        self.bar_n = len(bar_string)
        self.postfix = ['']*self.bar_n
        self.progress = [0]*self.bar_n
        self.total = [-1]*self.bar_n
        self.max_line = 0
        self.bar_l = l
    def update(self,i,progress=None,total=None,title = None):
        if progress is not None:
            self.progress[i] += progress
        if total is not None:
            self.total[i] = total
        if title is not None:
            self.bar_string[i] = title
    def init(self,i):
        self.progress[i] = 0
    def update_bar(self):
        self.refresh()
    def set_postfix_str(self,i,postfix):
        self.postfix[i] = postfix
    def refresh(self):
        text = '\r'
        for i in range(self.bar_n):
            if self.total[i] == -1:
                current_line = ''
            else:
                p = float(self.progress[i])/(self.total[i]+1e-6)
                if p>1:
                    p=1
                elif p<0:
                    p=0
                block = int(round(p*self.bar_l))
                current_line = "%s: %5.1f%%|%s| %d/%d %s"%( self.bar_string[i],p*100,"#"*block + "-"*(self.bar_l-block), self.progress[i],self.total[i],self.postfix[i])
                self.max_line = max(len(current_line),self.max_line)
            text += current_line + ' '*(self.max_line-len(current_line)) + '\n'
        text += '\033[%dA'%(self.bar_n)
        sys.stdout.write(text)
        sys.stdout.flush()
    def end(self):
        text = '\n'*self.bar_n
        sys.stdout.write(text)
        sys.stdout.flush()

def read_mapping(map_f,replace_prefix:List = None):
    map_dict = {}
    with open(map_f,'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[line[0]] = line[1].replace(replace_prefix[0],replace_prefix[1])
    return map_dict

#%%# Read sites from IP data
def read_yeast_sites(sites_f):
    putative_site_df = pd.read_excel(sites_f,engine='openpyxl')
    putative_site_start = putative_site_df["Peak genomic coordinate"].values - putative_site_df["Distance from peak to nearest RGAC site"].values
    putative_site_end = putative_site_df["Peak genomic coordinate"].values + putative_site_df["Distance from peak to nearest RGAC site"].values
    putative_ref = putative_site_df["Peak chr"].values
    putative_site_span = putative_site_df["Distance from peak to nearest RGAC site"].values
    putative_site_start_exact = putative_site_start[putative_site_span == 0]
    putative_site_end_exact = putative_site_end[putative_site_span == 0]
    putative_ref_exact = putative_ref[putative_site_span == 0]
    return putative_site_start_exact,putative_site_end_exact,putative_ref_exact

def read_human_sites(sites_f,filter_meth = None):
    putative_site_df = pd.read_csv(sites_f)
    putative_site_start = putative_site_df["genomic_position"].values
    putative_ref = putative_site_df["chr"].values
    #exclude nan and chr22_KI270734v1_random from pandas array
    mask = ~pd.isna(putative_ref) & (putative_ref != 'chr22_KI270734v1_random')
    if filter_meth:
        mask = mask & (putative_site_df["modification_status"] > 0.5)
    elif filter_meth == False:
        mask = mask & (putative_site_df["modification_status"] < 0.5)
    putative_site_start_exact = putative_site_start[mask]
    putative_site_end_exact = putative_site_start_exact
    putative_ref_exact = putative_ref[mask]
    return putative_site_start_exact,putative_site_end_exact,putative_ref_exact

def read_at_sites(sites_f, filter_meth = None):
    putative_site_df = pd.read_csv(sites_f)
    putative_site_start = putative_site_df["genomic_position"].values.astype(int)
    putative_ref = putative_site_df["chr"].values
    mask = ~pd.isna(putative_ref)
    if filter_meth:
        mask = mask & (putative_site_df["modification_status"] > 0.5)
    elif filter_meth == False:
        mask = mask & (putative_site_df["modification_status"] < 0.5)
    putative_site_start_exact = putative_site_start[mask]
    putative_site_end_exact = putative_site_start_exact
    putative_ref_exact = putative_ref[mask].astype(str)
    return putative_site_start_exact,putative_site_end_exact,putative_ref_exact

def transcript_position(genomics_contig,genomics_start,genomics_end,g2t_dict):
    """Transfer the genomics position to transcript position
    Args:
        genomics_contig (List): An array of genomics contigs
        genomics_start (List): An array of genomics start positions
        genomics_end (List): An array of genomics end positions
        g2t_dict (Dict): A nested dictionary mapping genomics position to transcript position
            key: (genomics_contig)
            value: A dictionary mapping genomics position to transcript position
                key: (genomics_position)
                value: A List of tuple [(t1_id,t1_pos),(t2_id,t2_pos),...]
    Returns:
        transcript_contig (List): An array of transcript contigs, the size of array may differ from the input
        transcript_start (List): An array of transcript start positions
        transcript_end (List): An array of transcript end positions
    """
    for ref,s,e in zip(genomics_contig,genomics_start,genomics_end):
        if "chr" in ref:
            ref = ref[3:]
        if ref in g2t_dict:
            if s in g2t_dict[ref] and e in g2t_dict[ref]:
                s_pos = g2t_dict[ref][s]
                e_ids = [i[0] for i in g2t_dict[ref][e]]
                for pos in s_pos:
                    if pos[0] in e_ids:
                        yield pos[0],pos[1],g2t_dict[ref][e][e_ids.index(pos[0])][1]
        
#%%
def collect_reads_from_bam(bam_f,
                           reference,
                           putative_site_start_exact,
                           putative_site_end_exact,
                           putative_ref_exact,
                           fast5_mapping,
                           span,
                           rep,
                           chrom = None,
                           pbar = None,
                           g2t_dict = None,
                           memory_saving = False):
    collections = {"rep":[],"ids":[],"fast5s":[],"seq_pos":[],"contig":[],
                   "refpos":[],"ref_segment":[],"segment":[],"query_seq":[],
                   "modified":[],"reverse_align":[],"basecall_seq":[],"qr_index":[],
                   "reference_seq":[]}
    bam = pysam.AlignmentFile(bam_f, "rb")
    reference = pysam.FastaFile(reference)
    skip_count = 0
    success_count = 0
    current = current_process()
    if len(current._identity):
        pid = current._identity[0]-1
    else:
        pid = 0
    if pbar is None:
        single_thread = True
        pbar = tqdm(total = len(putative_site_start_exact),position = 0, desc = f"{rep} {chrom if chrom else ''}")
    else:
        single_thread = False
        pbar.init(pid)
        pbar.set_postfix_str(pid,'')
        pbar.update(pid,total = len(putative_site_start_exact),title = f"#{pid} {rep} {chrom if chrom else ''}")
    if g2t_dict:
        pos_iterator = transcript_position(putative_ref_exact,putative_site_start_exact,putative_site_end_exact,g2t_dict)
    else:
        pos_iterator = zip(putative_ref_exact,putative_site_start_exact,putative_site_end_exact)
    for ref,s,e in pos_iterator:
        try:
            ref_len = reference.get_reference_length(ref)
        except KeyError:
            print(f"Reference {ref} not found in the reference file.")
            continue
        for read in bam.fetch(ref,max(s-1,0),min(e+1,ref_len)):
            if read.query_name not in fast5_mapping:
                skip_count += 1
                continue
            pos = np.asarray(read.get_aligned_pairs(),dtype = float)
            refseq = reference.fetch(ref,max(s-span-1,0),min(e+span,ref_len)).upper()
            pos = pos[~np.isnan(pos[:,1])]
            pos = pos[~np.isnan(pos[:,0])]
            q_idx = pos[pos[:,1] >= s-1,0][0] #s is 1-based
            collections['ids'].append(read.query_name)
            collections['rep'].append(rep)
            collections['segment'].append('')
            collections['fast5s'].append(fast5_mapping[read.query_name])
            collections['query_seq'].append(read.query_sequence)
            if not memory_saving:
                collections['basecall_seq'].append("") #commented out to save memory
                collections['reference_seq'].append(read.get_reference_sequence())
                collections['qr_index'].append(':'.join([str(int(x)) for x in pos[:,0]])+'|'+':'.join([str(int(x)) for x in pos[:,1]]))
            else:
                collections['basecall_seq'].append("*")
                collections['reference_seq'].append("*")
                collections['qr_index'].append("*")
            collections['contig'].append(ref)
            collections['seq_pos'].append(int(q_idx))
            collections['refpos'].append(s)
            collections['ref_segment'].append(refseq)
            collections['modified'].append(False)
            collections['reverse_align'].append(read.is_reverse)
            success_count += 1
        if single_thread:
            pbar.set_postfix_str(f"Success reads: {success_count}, Skip: {skip_count}")
            pbar.update()
        else:
            pbar.set_postfix_str(pid, f"Success reads: {success_count}, Skip: {skip_count}")
            pbar.update(pid,progress = 1)
            pbar.update_bar()
    if len(collections["rep"]) == 0:
        print(f"No read is succesfully collected for {rep} {chrom}, probably due to a incosistent genomeics2transcript mapping, please check if the contig name is consistent.")
    pbar.init(pid)
    bam.close()
    reference.close()
    return collections

def get_transcript_sites_from_genomics_position(genomics_pos,genomics_contig,g2t_dict):
    t_id,t_pos = [],[]
    entry = g2t_dict[genomics_contig]
    if genomics_pos not in entry.keys():
        return None,None
    for t in entry[genomics_pos]:
        t_id.append(t[0])
        t_pos.append(t[1])
    return t_id,t_pos

#%% read the xron output
def read_bc_from_fast5(collections,span,pbar = None):
    df = pd.DataFrame(collections)
    current = current_process()
    if len(current._identity):
        pid = current._identity[0]-1
    else:
        pid = 0
    if pbar is None:
        single_thread = True
        pbar = tqdm(total = len(df['ids']),position = 0, desc = f"read fast5")
    else:
        single_thread = False
        pbar.update(pid,total = len(df['ids']))
        pbar.set_postfix_str(pid, f"read basecall from fast5")
    for f in set(df['fast5s'].values):
        ids = df[df['fast5s'] == f]['ids'].values
        cols = df[df['fast5s'] == f].index
        with h5py.File(f,'r') as f5:
            for id,idx in zip(ids,cols):
                try:
                    result_h = f5[f"read_{id}"]['Analyses/Basecall_1D_001/BaseCalled_template/']
                except KeyError as e:
                    try:
                        result_h = f5[f"read_{id}"]['Analyses/Basecall_1D_000/BaseCalled_template/']
                    except KeyError as e:
                        continue
                seq = str(np.asarray(result_h['Fastq']).astype(str)).split('\n')[1]
                # collections['basecall_seq'][idx] = seq
                if collections['reverse_align'][idx]:
                    bc_idx = len(seq) - 1 - int(collections['seq_pos'][idx])
                else:
                    bc_idx = int(collections['seq_pos'][idx])
                collections['segment'][idx] = seq[max(bc_idx-span,0):bc_idx+span+1]
                collections['modified'][idx] = 'M' in collections['segment'][idx]
                if single_thread:
                    pbar.update(1)
                else:
                    pbar.update(pid,progress = 1)
                    pbar.update_bar()
    df = pd.DataFrame(collections)
    return df

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Select reads by sites')
    parser.add_argument('--threads', type=int, default=None, help='Number of threads')
    parser.add_argument('--run', type = str, required = True, help = "Which run to go, can be yeast or human")
    parser.add_argument('--model', type = str, required = True, help = "Which model output to collect from, e.g. xron_eneheke1")
    parser.add_argument('--names', type = str, default = None, help = "Names of the samples, split by comma")
    parser.add_argument('--reps', type = str, default = None, help = "Reps of the samples, split by comma, need to have the same length as names")
    parser.add_argument('--subset', type = str, default = None, help = "Which subset of sites to use can be balance,500genes,test. Default is None, which use all the sites.")
    parser.add_argument('--filter', type = bool, default = None, help = "If we want to filter the sites, default is None, all sites are selected.")
    parser.add_argument('--store_full', action = "store_ture", dest = "store_full", help = "Store the full output, default is False")
    args = parser.parse_args()
    args.names = args.names.split(',') if args.names else None
    args.reps = args.reps.split(',') if args.reps else None
    if args.reps:
        assert len(args.reps) == len(args.names), "The length of reps and names should be the same."
    return args

if __name__ =="__main__":
    import pickle
    #get folder path from environemnt variable
    scratch_f = os.environ['SCRATCH']+'/'
    args = parse_args()
    ### Yeast datasets base configure
    config_yeast = {
        'threads_n': 1 if args.threads is None else args.threads,
        'separate_chrom': False,
        'skip_processeed': False,
        'sites_f': f"{scratch_f}/Xron_Project/m6A_putative_site_from_RRACH_site_paper.xlsx",
        'organism': 'yeast',
        'assess_folder': "assess",
        'reference': f"{scratch_f}/ime4_Yearst/Yearst/Yeast_sk1.fasta",
        'base_f': f"{scratch_f}/ime4_Yearst/Yearst_full/",
        'g2t_dict': None,
    }
    ### Human datasets base configure
    config_human = {
        'threads_n': 47 if args.threads is None else args.threads,
        'separate_chrom': True,
        'skip_processeed': False,
        'sites_f': f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T.csv",
        'organism': 'human',
        'reference': f"{scratch_f}/NA12878_RNA_IVT/GRCh38_transcript_ensembel_sgnex/Homo_sapiens.GRCh38.cdna.ncrna.fa",
        'g2t_dict': f"{scratch_f}/Xron_Project/Benchmark/HEK293T/g2t_dict.pkl",
        'base_f': f"{scratch_f}/HEK293T/",
    }
    if args.subset == "balance":
        config_human['sites_f'] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_balanced.csv"
    elif args.subset.startswith("500"):
        config_human['sites_f'] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_subset500genes.csv"
    elif args.subset == "test":
        config_human['sites_f'] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_test.csv"

    ### Arabidopsis thaliana dataset base configure
    config_at = {
        'threads_n': 7 if args.threads is None else args.threads,
        'separate_chrom': True,
        'skip_processeed': False,
        'sites_f':f"{scratch_f}/Xron_Project/virc_test_results.csv",
        'organism': "thale_cress",
        'reference': f"{scratch_f}/TAIR10/Arabidopsis_thaliana.TAIR10.cdna.ncrna.fa",
        'g2t_dict': f"{scratch_f}/Xron_Project/Benchmark/AT/g2t_dict.pkl",
        'base_f': f"{scratch_f}/AT/",
    }
    if args.subset == "balance":
        config_at['sites_f'] = f"{scratch_f}/Xron_Project/virc_test_results_balanced.csv"
    elif args.subset.startswith("500"):
        config_at['sites_f'] = f"{scratch_f}/Xron_Project/virc_test_results_subset500genes.csv"
    elif args.subset == "test":
        config_at['sites_f'] = f"{scratch_f}/Xron_Project/virc_test_results_test.csv"
    

    #%% Yeast datasets configure 1  this is the configuration to generate Figure 2
    if args.run == "yeast_old":
        config = config_yeast
        config['repo'] = "xron_crosslink2_YeastFinetune_4000L"
        config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/Yeast/xron_crosslink2_finetune/"
        # config['repo'] = "xron_crosslink2_YeastFinetune_4000L"
        config['repo'] = "xron_crosslink2_YeastFinetune_4000L_OF"
        config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/Yeast/xron_crosslink2_finetune_OF/"
        config['reps'] = ["WT1_RNAAA023484","KO1_RNAAA024588","KO2_RNAAB058843","KO3_RNAAB059882","WT2_RNAAB056712","WT3_RNAAB057791"]
        config['names'] = ["wt1","ko1","ko2","ko3","wt2","wt3"]
        putative_site_start,putative_site_end,putative_ref = read_yeast_sites(config['sites_f'])

    #%% Human datasets
    if args.run == "human_old":
        config = config_human
        g2t_dict = config['g2t_dict']
        if g2t_dict:
            with open(g2t_dict,'rb') as f:
                g2t_dict = pickle.load(f)
        # config['sites_f] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_balanced.csv"; config["names"] = ["wt1-balance","ko1-balance"]
        # config['sites_f] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_subset500genes.csv"; config["names"] = ["wt1-500","ko1-500"]
        # config['sites_f] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_test.csv"; config["names"] = ["wt1-test","ko1-test"]
        config['sites_f'] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T.csv"; config["names"] = ["wt1"]; #config["names"] = ["wt1","ko1"]


        config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_finetune_positive_site/"; config['repo'] = "xron_crosslink2_YeastFinetune_4000L"; config['assess_folder'] = "assess_transcript"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L/"; repo = "xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L"; assess_folder = "assess_transcript"; 
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L_OF/"; repo = "xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L_OF" ;assess_folder = "assess_minimap2"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293T+MERGEFinetune_Converge_4000L/"; repo = "xron_crosslink2_Yeast+HEK293T+MERGEFinetune_Converge_4000L" ;assess_folder = "assess_minimap2"
        config['reps'] = ["HEK293T-WT-rep1"] # config['reps'] = ["HEK293T-WT-rep1","HEK293T-Mettl3-KO-rep1"]
        putative_site_start,putative_site_end,putative_ref = read_human_sites(config['sites_f'])

    #%% Yeast datasets configure 2 (Deprecated)
    # if args.run == "yeast":
    #     config = config_yeast
    #     # config['repo'] = "xron_crosslink2_YeastFinetune_4000L_OF"
    #     config['repo'] = "xron_crosslink2_Yeast+HEK293TFinetune_4000L"
    #     config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/Yeast/xron_crosslink2_Yeast+HEK293T_finetune/"
    #     config['reps'] = ["WT1_RNAAA023484","KO1_RNAAA024588","KO2_RNAAB058843","KO3_RNAAB059882","WT2_RNAAB056712","WT3_RNAAB057791"]
    #     config['names'] = ["wt1","ko1","ko2","ko3","wt2","wt3"]
    #     config['assess_folder'] = "assess_minimap2"
    #     putative_site_start,putative_site_end,putative_ref = read_yeast_sites(config['sites_f'])

    #%% Yeast datasets configure 2
    if args.run == "yeast":
        model = args.model
        config = config_yeast
        # config['repo'] = "xron_crosslink2_YeastFinetune_4000L_OF"
        config['repo'] = model
        config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/Yeast/{model}/"
        config['reps'] = ["WT1_RNAAA023484","KO1_RNAAA024588","KO2_RNAAB058843","KO3_RNAAB059882","WT2_RNAAB056712","WT3_RNAAB057791"]
        config['names'] = ["wt1","ko1","ko2","ko3","wt2","wt3"]
        config['assess_folder'] = "assess_minimap2"
        config['names'] = args.names if args.names else config['names']
        config['reps'] = args.reps if args.reps else config['reps']
        g2t_dict = config['g2t_dict']
        putative_site_start,putative_site_end,putative_ref = read_yeast_sites(config['sites_f'])

    #%% Human datasets
    if args.run == "human":
        config = config_human
        model = args.model
        g2t_dict = config['g2t_dict']
        if g2t_dict:
            with open(g2t_dict,'rb') as f:
                g2t_dict = pickle.load(f)
        # config['sites_f] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_balanced.csv"; config["names"] = ["wt1-balance","ko1-balance"]
        # config['sites_f] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_subset500genes.csv"; config["names"] = ["wt1-500","ko1-500"]
        # config['sites_f] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_test.csv"; config["names"] = ["wt1-test","ko1-test"]
        # config['sites_f'] = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T.csv"; config["names"] = ["wt1"]; #config["names"] = ["wt1","ko1"]


        # config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_finetune_positive_site/"; config['repo'] = "xron_crosslink2_YeastFinetune_4000L"; config['assess_folder'] = "assess_transcript"
        config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/{model}/"; config['repo'] = model; config['assess_folder'] = "assess_transcript"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L/"; repo = "xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L"; assess_folder = "assess_transcript"; 
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L_OF/"; repo = "xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L_OF" ;assess_folder = "assess_minimap2"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293T+MERGEFinetune_Converge_4000L/"; repo = "xron_crosslink2_Yeast+HEK293T+MERGEFinetune_Converge_4000L" ;assess_folder = "assess_minimap2"
        config['reps'] = ["HEK293T-WT-rep1"] # config['reps'] = ["HEK293T-WT-rep1","HEK293T-Mettl3-KO-rep1"]
        config['names'] = args.names if args.names else config['names']
        config['reps'] = args.reps if args.reps else config['reps']
        putative_site_start,putative_site_end,putative_ref = read_human_sites(config['sites_f'])

    #%% AT datasets
    if (args.run == "AT") or (args.run == "at") or (args.run == "Arabidopsis thaliana"):
        config = config_at
        model = args.model
        g2t_dict = config['g2t_dict']
        if g2t_dict:
            with open(g2t_dict,'rb') as f:
                g2t_dict = pickle.load(f)
        config["names"] = args.names if args.names else ["wt1","wt2","wt3"]
        config['out_f'] = f"{scratch_f}/Xron_Project/Benchmark/AT/{model}/"; config['repo'] = model; config['assess_folder'] = "assess_transcript"
        config['reps'] = args.reps if args.reps else ["col0_nanopore_drs_1","col0_nanopore_drs_2","col0_nanopore_drs_3"]
        putative_site_start,putative_site_end,putative_ref = read_at_sites(config['sites_f'])
        os.makedirs(config['out_f'],exist_ok = True)
    
    print(config)
    reference = config['reference']
    out_f = config['out_f']
    base_f = config['base_f']
    threads_n = config['threads_n']
    repo = config['repo']
    assess_folder = config['assess_folder']
    skip_processeed = config['skip_processeed']
    separate_chrom = config['separate_chrom']
    reps = config['reps']
    names = config['names']
    os.makedirs(out_f,exist_ok = True)

    #%% multiprocessing
    BaseManager.register('MultiPbars',MultiProcessBars)
    manager = BaseManager()
    manager.start()
    # multiprocessing with rep,n and chrom
    pbars = manager.MultiPbars(bar_string = ['']*(threads_n+1)) #Main process to keep the manager
    def worker(arguments):
        time.sleep(random.random())
        rep,n,chrom,pbars = arguments
        if chrom != None:
            putative_site_start_exact = putative_site_start[putative_ref == chrom]
            putative_site_end_exact = putative_site_end[putative_ref == chrom]
            putative_ref_exact = putative_ref[putative_ref == chrom]
            name = n+"_"+chrom
        else:
            putative_site_start_exact = putative_site_start
            putative_site_end_exact = putative_site_end
            putative_ref_exact = putative_ref
            name = n
        #check if output already exist
        if os.path.exists(os.path.join(out_f,f"readIDs_{repo}_{name}.csv")) and skip_processeed:    
            print(f"Output already exist for {name}! Skip...")
            return
        assess_f = f"{base_f}/{rep}/{repo}/{assess_folder}"
        bam_f = f"{assess_f}/aln.sorted.bam"
        span = 5
        if threads_n == 1:
            print(f"Total {len(putative_ref_exact)} DRACH reference sites.")
        fast5_mapping = read_mapping(f"{assess_f}/merge.fastq.index",replace_prefix=["/ocean/projects/hmcmutc/haotiant/",f"{scratch_f}"])
        if threads_n == 1:
            print("Collecting reads from bam...")
        current = current_process()
        id = current._identity
        if len(id) == 0:
            pos = 0
        else:
            pos = id[0]-1
        # pbar = tqdm(total = len(putative_ref_exact),position = pos, desc = f"#{pos} read bam: {rep} {chrom}")
        collections = collect_reads_from_bam(f"{bam_f}",
                                            reference,
                                            putative_site_start_exact,
                                            putative_site_end_exact,
                                            putative_ref_exact,
                                            fast5_mapping,
                                            span = span,
                                            rep = rep,
                                            chrom = chrom,
                                            pbar = pbars,
                                            g2t_dict=g2t_dict,
                                            memory_saving = (not args.store_full))
        if threads_n == 1:
            print("Collecting info from fast5...")
        # pbar = tqdm(total = len(collections['ids']),position = pos, desc = f"#{pos} read fast5:{rep} {chrom}")
        df = read_bc_from_fast5(collections,span = span,pbar = pbars)
        df.to_csv(os.path.join(out_f,f"readIDs_{repo}_{name}.csv"))
    # generate iterator of rep,n and chrom
    if separate_chrom:
        running_args = [(rep,n,chrom,pbars) for rep,n in zip(reps,names) for chrom in np.unique(putative_ref)]
    else:
        running_args = [(rep,n,None,pbars) for rep,n in zip(reps,names)]
    # multiprocessing
    if threads_n > 1:
        with Pool(threads_n) as p:
            p.map(worker,running_args)
    else:
        for arg in running_args:
            print(f"Processing {' '.join(arg[:2])}...")
            worker(arg)
