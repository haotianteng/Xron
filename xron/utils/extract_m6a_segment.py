#This script is used to prepare training dataset for methylation basecaller by extracting raw signal chunks with m6a segments
import re
import os
import h5py
import pysam
import editdistance
from Bio import motifs
from Bio.Seq import Seq 
from sklearn.mixture import GaussianMixture
from xron.utils.seq_op import norm_by_noisiest_section
import numpy as np
import pandas as pd
from tqdm import tqdm

motif_reg = {"A":"A","C":"C","G":"G","T":"T","U":"U","R":"[AG]","Y":"[CT]","M":"[AC]","K":"[GT]","S":"[GC]","W":"[AT]","B":"[CGT]","D":"[AGT]","H":"[ACT]","V":"[ACG]","N":"[ACGT]"}
def get_reg(motif):
    reg = ""
    for i in motif:
        reg += motif_reg[i]
    return reg

def rc(seq):
    seq = seq.upper()
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join([complement[base] for base in reversed(seq)])

def parse_qr_string(qr_string):
    q,r = qr_string.split('|')
    q = np.asarray([int(float(x)) for x in q.split(':')])
    r = np.asarray([int(float(x)) for x in r.split(':')])
    # r = r-r[0] #Make the first index 0
    return q,r

def exclude_unmached_read(df):
    """Exclude reads whose basecall sequence is not the same as the query sequence
    This is usually due to there is repeating read name id.
    """
    dist = np.asarray([editdistance.eval(rc(x),y) if z else editdistance.eval(x,y) for x,y,z in tqdm(zip(df['query_seq'].values,df['basecall_seq'].values,df['reverse_align'].values))])
    gmm = GaussianMixture(n_components=2)
    gmm.fit(dist.reshape(-1,1))
    cluster_means = gmm.means_
    threshold = np.max([np.percentile(cluster_means,95),np.mean(cluster_means)])
    mask = dist < threshold
    df = df[mask]
    return df

def get_ref_idx(q,r,q_idx):
    return r[np.where(q>=q_idx)[0][0]]

#Set a repository name in the system environment variable NAME
CONFIG = "HEK293T"
BRIDGE_FOLDER = os.environ['SCRATCH']+'/'
if CONFIG == "Yearst":
    repo = "xron_crosslink2"
    names = ["ko1","ko2","ko3","wt1","wt2","wt3"]
    meths = [False,False,False,True,True,True]
    reference = f"{BRIDGE_FOLDER}/ime4_Yearst/Yearst/Yeast_sk1.fasta"
    reference = pysam.FastaFile(reference)
    segment_len = 4000
    window_half = 4
    xron_basecall_config = {"stride":11,
                            "move_forward":False,}
    motif_string = "DRACD"
    motif_A_idx = 2
    BASE_F = f"{BRIDGE_FOLDER}/Xron_Project/Benchmark/Yearst"

if CONFIG == "HEK293T":
    repo = "xron_crosslink2_YeastFinetune_4000L"
    # names = ["wt1-balance_chr1",
    #          "wt1-balance_chr2",
    #          "wt1-balance_chr3",
    #          "wt1-balance_chr4",
    #          "wt1-balance_chr5",
    #          "wt1-balance_chr6",
    #          "wt1-balance_chr7",
    #          "wt1-balance_chr8",
    #          "wt1-balance_chr9",
    #          "wt1-balance_chr10",
    #          "wt1-balance_chr11",
    #          "wt1-balance_chr12",
    #          "wt1-balance_chr13",
    #          "wt1-balance_chr14",
    #          "wt1-balance_chr15",
    #          "wt1-balance_chr16",
    #          "wt1-balance_chr17",
    #          "wt1-balance_chr18",
    #          "wt1-balance_chr19",
    #          "wt1-balance_chr20",
    #          "wt1-balance_chr21",
    #          "wt1-balance_chr22",
    #          "wt1-balance_chrX",
    #          "wt1-balance_chrM",]
    # names = ["wt1-balance_chr3",
    #          "wt1-balance_chr4",
    #          "wt1-balance_chr5",
    #          "wt1-balance_chr6",
    #          "wt1-balance_chr7",
    #          "wt1-balance_chr8",
    #          "wt1-balance_chr9",
    #          "wt1-balance_chr10",
    #          "wt1-balance_chr11",
    #          "wt1-balance_chr12",
    #          "wt1-balance_chr13"]
    # names = ["wt1-balance_chr14",
    #          "wt1-balance_chr15",
    #          "wt1-balance_chr16",
    #          "wt1-balance_chr17",
    #          "wt1-balance_chr18",
    #          "wt1-balance_chr19",
    #          "wt1-balance_chr20",
    #          "wt1-balance_chr21",
    #          "wt1-balance_chr22",
    #          "wt1-balance_chrX",
    #          "wt1-balance_chrM"]
    names = [os.environ['NAME']]
    #TODO change the name
    meths = [None]*len(names)
    BASE_F = f"{BRIDGE_FOLDER}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_finetune/"
    sites_f = f"{BRIDGE_FOLDER}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T.csv"
    sites = pd.read_csv(sites_f)
    reference = f"{BRIDGE_FOLDER}/NA12878_RNA_IVT/GRCh38_transcript_ensembel_sgnex/Homo_sapiens.GRCh38.cdna.ncrna.fa"
    reference = pysam.FastaFile(reference)
    segment_len = 4000
    window_half = 4
    xron_basecall_config = {"stride":11,
                            "move_forward":False,}
    motif_string = "DRACD"
    motif_A_idx = 2


for name,meth in zip(names,meths):
    input_site_summary = f"{BASE_F}/readIDs_{repo}_{name}.csv"
    out_f = f"{BASE_F}/datasets/{repo}_{name}_{motif_string}/"
    os.makedirs(out_f,exist_ok=True)
    print(f"Loading summary {name}...")
    config = xron_basecall_config
    df = pd.read_csv(input_site_summary)
    df = df.dropna()
    print("Excluding unmatched reads...")
    df = exclude_unmached_read(df)
    print("Loading reference...")
    chunks = []
    seqs = []
    chunk_lens = []
    seq_lens = []
    motif_segments = []
    summary = {"succeed":0,"Broken basecall entry":0, "Sampling sequence out of alignment bounds":0,"Motif is missing":0}

    if type(motif_string) is str:
        motif = re.compile(get_reg(motif_string))
    print("Extracting segments...")
    with tqdm(total = len(df)) as pbar:
        for f in set(df['fast5s'].values):
            ids = df[df['fast5s'] == f]['ids'].values
            cols = df[df['fast5s'] == f].index
            with h5py.File(f.replace("/ocean/projects/hmcmutc/haotiant/",f"{BRIDGE_FOLDER}"),'r') as f5:
                for id,idx in zip(ids,cols):
                    q,r = parse_qr_string(df['qr_index'][idx])
                    refpos = df['refpos'][idx]-1
                    read_h = f5[f"read_{id}"]
                    result_h = read_h['Analyses/Basecall_1D_000/BaseCalled_template/']
                    signal = np.asarray(read_h['Raw/Signal']) 
                    original_signal = signal.astype(np.float32)
                    signal,med,mad = norm_by_noisiest_section(signal)
                    signal = signal.astype(np.float16)[::-1]#Reverse the signal because RNA is 5'->3'
                    seq = str(np.asarray(result_h['Fastq']).astype(str)).split('\n')[1]
                    move = np.asarray(result_h['Move'])
                    if len(move) == 0:
                        summary["Broken basecall entry"] += 1
                        pbar.update()
                        continue
                    if config['move_forward']:
                        move = move[::-1]

                    #build the signal index of the sequence with 11 increase step from move matrix
                    signal_idx = np.where(move == 1)[0]*config['stride']

                    #Sample a signal segment of 4000 length around the m6A site
                    upstream = np.random.randint(0,segment_len - 200)
                    q_seq = df['query_seq'][idx]
                    q_pos = df['seq_pos'][idx]
                    reverse = df['reverse_align'][idx]
                    bc_pos = q_pos if not reverse else len(q_seq) - q_pos
                    try:
                        start = max(signal_idx[bc_pos] - upstream,0)
                    except IndexError:
                        summary["Broken basecall entry"] += 1
                        pbar.update()
                        continue
                    bc_start,bc_end = np.where(signal_idx>=start)[0][0],np.where(signal_idx<=start+segment_len)[0][-1]
                    q_start = bc_start if not reverse else len(q_seq) - bc_end
                    q_end = bc_end if not reverse else len(q_seq) - bc_start
                    if q_start - q[0] < -window_half or q_end - q[-1] > window_half:
                        summary["Sampling sequence out of alignment bounds"] += 1
                        pbar.update()
                        continue
                    q_start = np.max([q_start,q[0]])
                    q_end = np.min([q_end,q[-1]])
                    ref_start = min(get_ref_idx(q,r,q_start),refpos-1)
                    ref_end = max(get_ref_idx(q,r,q_end),refpos+1)
                    if abs(ref_end - ref_start - (q_end - q_start)) > 10:
                        summary["Sampling sequence out of alignment bounds"] += 1
                        pbar.update()
                        continue
                    try:
                        ref_seq = reference.fetch(df['contig'][idx],ref_start,ref_end+1)
                    except ValueError as e:
                        print(e)
                        summary["Sampling sequence out of alignment bounds"] += 1
                        pbar.update()
                        continue
                    refpos = refpos - ref_start #Relative position to the start of the segment
                    if reverse:
                        ref_seq = rc(ref_seq)
                        refpos = len(ref_seq) - refpos
                    if ref_seq[refpos] != 'A':
                        motif_hit = re.search(motif,ref_seq[refpos-window_half:refpos+window_half+1])
                        if not motif_hit:
                            summary["Motif is missing"] += 1
                            pbar.update()
                            continue
                        else:
                            refpos += motif_hit.start() - window_half + motif_A_idx #Fixing some minor shiftings
                    motif_segments.append(ref_seq[refpos-window_half:refpos+window_half+1])
                    if meth is not None:
                        if meth:
                            #replace A with M at refpos to indicate the methylation
                            ref_seq = ref_seq[:refpos] + 'M' + ref_seq[refpos+1:]
                    else:
                        entry = sites[(sites['chr'] == df['contig'][idx]) & (sites['genomic_position'] == df['refpos'][idx])]
                        if len(entry) == 0:
                            summary["Motif is missing"] += 1
                            continue
                        if entry['modification_status'].values[0] == 1:
                            ref_seq = ref_seq[:refpos] + 'M' + ref_seq[refpos+1:]
                    seq_lens.append(len(ref_seq))
                    seqs.append(ref_seq)
                    curr_seg = signal[start:start+segment_len]
                    chunk_lens.append(len(curr_seg))
                    if len(curr_seg) < segment_len:
                        curr_seg = np.pad(curr_seg,(0,segment_len-len(curr_seg)),'constant',constant_values=0)
                    chunks.append(curr_seg)
                    summary["succeed"] += 1
                    pbar.set_postfix({"succeed":summary["succeed"],
                                    "Broken basecall entry":summary["Broken basecall entry"],
                                    "Sampling sequence out of alignment bounds":summary["Sampling sequence out of alignment bounds"],
                                    "Motif is missing":summary["Motif is missing"]})
                    pbar.update()
    print("Saving...")
    np.save(out_f + "seqs.npy",seqs)
    np.save(out_f + "chunk_lens.npy",chunk_lens)
    np.save(out_f + "seq_lens.npy",seq_lens)
    np.save(out_f + "chunks.npy",np.asarray(chunks))
    motif_segments = [Seq(x) for x in motif_segments if len(x) == (window_half*2+1) ]
    motif = motifs.create(motif_segments)
    print("Done!")
