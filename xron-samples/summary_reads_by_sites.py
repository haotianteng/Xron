import os
import pysam
from tqdm import tqdm
import pandas as pd
import numpy as np
#%%# Summary site level data
def geo_mean_overflow(iterable,epsilon = 1e-6):
    if len(iterable) == 0:
        return 0
    return np.exp(np.log(iterable+1e-6).mean())

def read_sites(sites_f):
    putative_site_df = pd.read_excel(sites_f,engine='openpyxl')
    putative_site_start = putative_site_df["Peak genomic coordinate"].values - putative_site_df["Distance from peak to nearest RGAC site"].values
    putative_site_end = putative_site_df["Peak genomic coordinate"].values + putative_site_df["Distance from peak to nearest RGAC site"].values
    putative_ref = putative_site_df["Peak chr"].values
    putative_site_span = putative_site_df["Distance from peak to nearest RGAC site"].values
    putative_site_start_exact = putative_site_start[putative_site_span == 0]
    putative_site_end_exact = putative_site_end[putative_site_span == 0]
    putative_ref_exact = putative_ref[putative_site_span == 0]
    return putative_site_start_exact,putative_site_end_exact,putative_ref_exact

def read_human_sites(sites_f):
    putative_site_df = pd.read_csv(sites_f)
    putative_site_start = putative_site_df["genomic_position"].values
    putative_ref = putative_site_df["chr"].values
    #exclude nan and chr22_KI270734v1_random from pandas array
    putative_site_start_exact = putative_site_start[~pd.isna(putative_ref) & (putative_ref != 'chr22_KI270734v1_random')]
    putative_site_end_exact = putative_site_start_exact
    putative_ref_exact = putative_ref[~pd.isna(putative_ref) & (putative_ref != 'chr22_KI270734v1_random')]
    return putative_site_start_exact,putative_site_end_exact,putative_ref_exact.astype(str)

def get_transcription_sites(ref,pos,g2t_dict):
    """Get all the possible transcription sites for a given genomics position
    Args:
        ref: reference name
        pos: genomic position
        g2t_dict: dictionary of genomic to transcription sites
    Returns:
        transcription sites: list of transcription sites with tuple of (transcription ref,transcription pos)
    """

    if "chr" in ref:
        ref = ref[3:]
    try: 
        return g2t_dict[ref][pos]
    except KeyError:
        return []

def summary_sites(df,
                  sites_f,
                  span,
                  reference,
                  g2t_dict = None,):
    if sites_f.endswith('.xlsx'):
        putative_site_start_exact,putative_site_end_exact,putative_ref_exact = read_sites(sites_f)
    else:
        putative_site_start_exact,putative_site_end_exact,putative_ref_exact = read_human_sites(sites_f)
    # sites_df = pd.read_csv(sites_f)
    print("Building index for input dataframe...")
    df.set_index(['contig','refpos'],inplace=True)
    reference = pysam.FastaFile(reference)
    summary = {"contig":[],"refpos":[],"ref_segment":[],"modified_prob":[],"coverage":[],"modified_prob_geometric":[],"read_prob_array":[]}
    with tqdm(total = len(putative_site_start_exact)) as pbar:
        for ref,s,e in zip(putative_ref_exact,putative_site_start_exact,putative_site_end_exact):
            # site_entry = sites_df[(sites_df['genomic_position'] == s) & (sites_df['chr'] == ref)]
            ref_len = reference.get_reference_length(str(ref))
            ref_segment = reference.fetch(ref,max(s-span,0),min(e+span-1,ref_len)).upper()
            # if ref_segment[span-1] != 'A':
            #     pbar.update(1)
            #     continue
            summary['contig'].append(ref)
            summary['refpos'].append(s)
            summary['ref_segment'].append(ref_segment)
            if g2t_dict is not None:
                transcription_sites = get_transcription_sites(ref,s,g2t_dict)
                if len(transcription_sites) == 0:
                    summary['modified_prob'].append(0)
                    summary['modified_prob_geometric'].append(0)
                    summary['read_prob_array'].append([])
                    summary['coverage'].append(0)
                    pbar.update(1)
                    continue
                modified_probs = []
                cum_coverage = 0
                for t_ref,t_pos in transcription_sites:
                    try:
                        hits = df.loc[(t_ref,t_pos)]
                        hits = hits[hits['reverse_align'] == False]
                        coverage = len(hits)
                        cum_coverage += coverage
                        modified_probs += hits['modified'].values.tolist()
                    except KeyError:
                        #no hit found
                        coverage = 0
                        continue
                modified_probs = np.array(modified_probs)
                summary['modified_prob'].append(np.mean(modified_probs))
                if len(modified_probs) == 0:
                    summary['modified_prob_geometric'].append(0)
                else:
                    summary['modified_prob_geometric'].append(1 - geo_mean_overflow(1-modified_probs))
                summary['read_prob_array'].append(modified_probs)
                summary['coverage'].append(cum_coverage)
                pbar.update(1)
                continue
            else:
                try:
                    hits = df.loc[(ref,s)]
                    hits = hits[hits['reverse_align'] == False]
                except KeyError:
                    #no hit found
                    summary['modified_prob'].append(np.nan)
                    summary['modified_prob_geometric'].append(0)
                    summary['read_prob_array'].append([])
                    summary['coverage'].append(0)
                    pbar.update(1)
                    continue
                coverage = len(hits)
                modified_probs = hits['modified'].values
                summary['modified_prob'].append(np.mean(modified_probs))
                if len(modified_probs) == 0:
                    summary['modified_prob_geometric'].append(0)
                else:
                    summary['modified_prob_geometric'].append(1 - geo_mean_overflow(1-modified_probs))
                summary['read_prob_array'].append(modified_probs)
                summary['coverage'].append(coverage)
                pbar.update(1)
    n_covered_sites = np.sum(np.array(summary['coverage']) > 0)
    print(f"Summarize {len(putative_site_start_exact)} sites, got {n_covered_sites} >0 coverage sites.")
    df_summary = pd.DataFrame(summary)
    return df_summary

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Select reads by sites')
    parser.add_argument('--run', type = str, required = True, help = "Which run to go, can be yeast or human")
    parser.add_argument('--input', type = str, required = True, help = "Which model output to collect from, e.g. xron_eneheke1")
    parser.add_argument('--names', type = str, default = None, help = "Names of the samples, split by comma")
    parser.add_argument('--subset', type = str, default = None, help = "Which subset of sites to use can be balance,500genes,test. Default is None, which use all the sites.")
    args = parser.parse_args()
    args.names = args.names.split(',') if args.names else None
    return args

if __name__ == "__main__":
    import pickle
    scratch_f = os.environ['SCRATCH']+'/'
    args = parse_args()
    #%% Yeast summary
    if args.run == "yeast":
        sites_f = f"{scratch_f}/Xron_Project/m6A_putative_site_from_RRACH_site_paper.xlsx"
        reference = f"{scratch_f}/ime4_Yearst/Yearst/Yeast_sk1.fasta"
        out_f = f"{scratch_f}/Xron_Project/Benchmark/Yeast/xron_crosslink2_finetune_OF/"
        repo = "xron_crosslink2_YeastFinetune_4000L_OF"
        if args.input:
            out_f = f"{scratch_f}/Xron_Project/Benchmark/Yeast/{args.input}/" ;repo = args.input
        os.makedirs(out_f,exist_ok = True)
        names = ["wt1","ko1","ko2","ko3","wt2","wt3"]
        names = args.names if args.names else names
        putative_site_start_exact,putative_site_end_exact,putative_ref_exact = read_sites(sites_f)
        span = 5
        for name in names:
            print(f"Summarize {name} to summary_{repo}_{name}.csv")
            df = pd.read_csv(os.path.join(out_f,f"readIDs_{repo}_{name}.csv"))
            df_summary = summary_sites(df,sites_f,span,reference)
            df_summary.to_csv(os.path.join(out_f,f"summary_{repo}_{name}.csv"))

    # #%% Human datasets
    if args.run == "human":
        if args.subset == "balance":
            sites_f = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_balanced.csv"; 
            names = ["wt1-balance","ko1-balance"]
        elif args.subset == "500genes":
            sites_f = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_subset500genes.csv"; 
            names = ["wt1-500"]; # names = ["wt1-500","ko1-500"];
        elif args.subset == "test":
            sites_f = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T_test.csv"; 
            names = ["wt1-test","ko1-test"]
        else:
            sites_f = f"{scratch_f}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T.csv"; 
            names = ["wt1"]; # names = ["wt1","ko1"]
        names = args.names if args.names else names
        reference = f"{scratch_f}/NA12878_RNA_IVT/GRCh38_sgnex/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"
        # reference = f"{scratch_f}/NA12878_RNA_IVT/GRCh38_transcript_ensembel_sgnex/Homo_sapiens.GRCh38.cdna.ncrna.fa" #Sgnex transcript reference which is used in m6anet
        
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_finetune/" ;repo = "xron_crosslink2_YeastFinetune_4000L"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_finetune_OF/"; repo = "xron_crosslink2_YeastFinetune_4000L_OF"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293T_finetune/" ;repo = "xron_crosslink2_Yeast+HEK293TFinetune_4000L"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L/" ;repo = "xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L"
        # out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L_OF/"; repo = "xron_crosslink2_Yeast+HEK293TFinetune_Converge_4000L_OF"
        out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/xron_eneheke1/" ;repo = "xron_eneheke1"
        if args.input:
            out_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/{args.input}/" ;repo = args.input
        print("Loading genome to transcript dictionary...")
        g2t_dict_f = f"{scratch_f}/Xron_Project/Benchmark/HEK293T/g2t_dict.pkl"
        with open(g2t_dict_f,'rb') as f:
            g2t_dict = pickle.load(f)

        os.makedirs(out_f,exist_ok = True)
        chromes = ['chr%d'%i for i in np.arange (1,23)] + ['chrX','chrM']
        # chromes = ['chr%d'%i for i in np.arange (1,3)]
        span = 5
        for name in names:
            print(f"Summarize {name} to summary_{repo}_{name}.csv")
            dfs = []
            for chr in chromes:
                print("Reading "+os.path.join(out_f,f"readIDs_{repo}_{name}_{chr}.csv"))
                if os.path.exists(os.path.join(out_f,f"readIDs_{repo}_{name}_{chr}.csv")):
                    dfs.append(pd.read_csv(os.path.join(out_f,f"readIDs_{repo}_{name}_{chr}.csv")))
            df = pd.concat(dfs)
            df_summary = summary_sites(df,sites_f,span,reference,g2t_dict)
            df_summary.to_csv(os.path.join(out_f,f"summary_{repo}_{name}.csv"))
    
    #%% AT dataset
    if args.run == "AT":
        if args.subset == "balance":
            sites_f = f"{scratch_f}/Xron_Project/virc_test_results_balanced.csv"
        elif args.subset == "500genes":
            sites_f = f"{scratch_f}/Xron_Project/virc_test_results_subset500genes.csv"
        elif args.subset == "test":
            sites_f = f"{scratch_f}/Xron_Project/virc_test_results_test.csv"
        else:
            sites_f = f"{scratch_f}/Xron_Project/virc_test_results.csv"
        names = args.names if args.names is not None else ["wt1", "wt2", "wt3"]
        reference = f"{scratch_f}/TAIR10/Arabidopsis_thaliana.TAIR10.dna.fa"
        
        out_f = f"{scratch_f}/Xron_Project/Benchmark/AT/xron_eneheke1/" ;repo = "xron_eneheke1"
        if args.input:
            out_f = f"{scratch_f}//Xron_Project/Benchmark/AT/{args.input}/" ;repo = args.input
        print("Loading genome to transcript dictionary...")
        g2t_dict_f = f"{scratch_f}/Xron_Project/Benchmark/AT/g2t_dict.pkl"
        with open(g2t_dict_f,'rb') as f:
            g2t_dict = pickle.load(f)

        os.makedirs(out_f,exist_ok = True)
        chromes = ['%d'%i for i in np.arange (1,5)] + ['Pt','Mt']
        # chromes = ['chr%d'%i for i in np.arange (1,3)]
        span = 5
        for name in names:
            print(f"Summarize {name} to summary_{repo}_{name}.csv")
            dfs = []
            for chr in chromes:
                print("Reading "+os.path.join(out_f,f"readIDs_{repo}_{name}_{chr}.csv"))
                if os.path.exists(os.path.join(out_f,f"readIDs_{repo}_{name}_{chr}.csv")):
                    dfs.append(pd.read_csv(os.path.join(out_f,f"readIDs_{repo}_{name}_{chr}.csv")))
            df = pd.concat(dfs)
            df_summary = summary_sites(df,sites_f,span,reference,g2t_dict)
            df_summary.to_csv(os.path.join(out_f,f"summary_{repo}_{name}.csv"))