# This script is written to include the operations including the coordinates mapping and transfer.
def reverse_mapping(t2g_dict):
    """Get the g2t_dict from t2g_dict, mapping genomics position to transcript position, 
    note that each genomics position can be mapped to multiple transcript positions.
    param t2g_dict[Dict]: transcript to genomics position dictionary
        key: transcript id
        value: a dict containing the mapping from transcript position to genomics position
            key: transcript position
            value: a tuple of (tx_contig, g_id, g_pos)
    return g2t_dict[Dict]: genomics to transcript position dictionary
        key: genomics contig
        value: a dictionary of genomics position 
            key: genomics position
            value: a list of tuple [(t1_id, t1_pos),(t2_id, t2_pos),...] containing all the possible transcript positions
    """
    g2t_dict = {}
    for t_id,t in t2g_dict.items():
        for t_pos, g in t.items():
            if g[0] not in g2t_dict:
                g2t_dict[g[0]] = {}
            if g[2] not in g2t_dict[g[0]]:
                g2t_dict[g[0]][g[2]] = []
            g2t_dict[g[0]][g[2]].append((t_id, t_pos))
    return g2t_dict

def t2g(tx_id, fasta_dict, gtf_dict):
    t2g_dict = {}
    if tx_id not in fasta_dict.keys():
        return [], []
    tx_seq = fasta_dict[tx_id]
    tx_contig = gtf_dict[tx_id]['chr']
    g_id = gtf_dict[tx_id]['g_id']
    if tx_seq is None:
        return [], []

    for exon_num in range(len(gtf_dict[tx_id]['exon'])):
        g_interval = gtf_dict[tx_id]['exon'][exon_num]
        tx_interval = gtf_dict[tx_id]['tx_exon'][exon_num]
        for g_pos in range(g_interval[0], g_interval[1] + 1): # Exclude the rims of exons.
            dis_from_start = g_pos - g_interval[0]
            if gtf_dict[tx_id]['strand'] == "+":
                tx_pos = tx_interval[0] + dis_from_start
            elif gtf_dict[tx_id]['strand'] == "-":
                tx_pos = tx_interval[1] - dis_from_start
            if (g_interval[0] <= g_pos < g_interval[0]+2) or (g_interval[1]-2 < g_pos <= g_interval[1]): 
                kmer = 'XXXXX'
            else:
                kmer = tx_seq[tx_pos-2:tx_pos+3]
            t2g_dict[tx_pos] = (tx_contig, g_id, g_pos) # tx.contig is chromosome.
    return t2g_dict

def readFasta(transcript_fasta_paths_or_urls):
    fasta=open(transcript_fasta_paths_or_urls,"r")
    entries=""
    for ln in fasta:
        entries+=ln
    entries=entries.split(">")
    dict={}
    for entry in entries:
        entry=entry.split("\n")

        if len(entry[0].split()) > 0:
            id=entry[0].split()[0]
            seq="".join(entry[1:])
            dict[id]=seq
    return dict

def readGTF(gtf_path_or_url):
    gtf=open(gtf_path_or_url,"r")
    dict={}
    gene_transcript_dict={}
    for ln in gtf:
        if not ln.startswith("#"):
            ln=ln.strip("\n").split("\t")
            if ln[2] == "transcript" or ln[2] in ("exon", "start_codon", "stop_codon", "CDS", 
                                                  "three_prime_utr", "five_prime_utr"):
                chr,type,start,end=ln[0],ln[2],int(ln[3]),int(ln[4])
                attrList=ln[-1].split(";")
                attrDict={}
                for k in attrList:
                    p=k.strip().split(" ")
                    if len(p) == 2:
                        attrDict[p[0]]=p[1].strip('\"')
                tx_id = attrDict["transcript_id"] + "." + attrDict["transcript_version"]
                g_id = attrDict["gene_id"]
                if g_id not in gene_transcript_dict:
                    gene_transcript_dict[g_id] = [tx_id]
                else:
                    gene_transcript_dict[g_id].append(tx_id)
                if tx_id not in dict:
                    dict[tx_id]={'chr':chr,'g_id':g_id,'strand':ln[6]}
                    if type not in dict[tx_id]:
                        if type == "transcript":
                            dict[tx_id][type]=(start,end)
                else:
                    if type == 'CDS':
                        info = (start, end, int(attrDict['exon_number']))
                    else:
                        info = (start, end)
                    if type not in dict[tx_id]:
                        dict[tx_id][type]=[info]
                    else:
                        dict[tx_id][type].append(info)
                          
    #convert genomic positions to tx positions
    for id in dict:
        tx_pos,tx_start=[],0
        for pair in dict[id]["exon"]:
            tx_end=pair[1]-pair[0]+tx_start
            tx_pos.append((tx_start,tx_end))
            tx_start=tx_end+1
        dict[id]['tx_exon']=tx_pos
    return dict,gene_transcript_dict

def map_nm_to_enst(nm_id):
    # Step 1: Retrieve the RefSeq accession associated with the NM ID using NCBI E-utilities API
    ncbi_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ncbi_params = {
        "db": "nucleotide",
        "id": nm_id,
        "rettype": "acc",
        "retmode": "text"
    }
    ncbi_response = requests.get(ncbi_url, params=ncbi_params)
    refseq_acc = ncbi_response.text.strip()

    # Step 2: Map the RefSeq accession to an Ensembl transcript ID using Ensembl REST API
    ensembl_url = "https://rest.ensembl.org/map/cdna/" + refseq_acc + "/GRCh38"
    ensembl_headers = {"Content-Type": "application/json"}
    ensembl_response = requests.get(ensembl_url, headers=ensembl_headers)
    ensembl_json = ensembl_response.json()
    if "mappings" in ensembl_json and len(ensembl_json["mappings"]) > 0:
        ensembl_id = ensembl_json["mappings"][0]["to_id"]
        return ensembl_id
    else:
        return None
    
if __name__ == "__main__":
    #%% Set up paths
    import os
    import pandas as pd
    import pickle
    from itertools import chain
    SCRATCH = os.environ['SCRATCH']
    ref_folder = os.path.join(SCRATCH, 'NA12878_RNA_IVT/GRCh38_transcript_ensembel_sgnex')
    trascript_f = os.path.join(ref_folder,'Homo_sapiens.GRCh38.cdna.ncrna.fa')
    gtf_f = os.path.join(ref_folder,'Homo_sapiens.GRCh38.91.gtf')
    sites_f = f"{SCRATCH}/Xron_Project/m6A_site_m6Anet_DRACH_HEK293T.csv"

    #%%Test case 1:
    #Build t2g dictionary for human reference from SG-NEX data
    out_f = f"{SCRATCH}/Xron_Project/Benchmark/HEK293T/t2g_dict.pkl"
    print("Reading reference fasta files...")   
    fasta = readFasta(trascript_f)
    print("Reading reference gtf files...")
    gtf,id_map = readGTF(gtf_f)
    print("Reading transcript names...")
    df = pd.read_csv(sites_f)
    gene_ids = df['gene_id'].unique()
    tx_names = [id_map[gene_id] for gene_id in gene_ids if gene_id in id_map]
    tx_names = list(chain(*tx_names)) #flatten the list
    print("Creating t2g dictionary...")
    t2g_dict = {tx:t2g(tx, fasta, gtf) for tx in tx_names}
    print("Saving t2g dictionary...")
    with open(out_f, "wb+") as f:
        pickle.dump(t2g_dict, f)

    #%%Test case 2:
    #Build reverse g2t dictionary for human reference 
    t2g_f = f"{SCRATCH}/Xron_Project/Benchmark/HEK293T/t2g_dict.pkl"
    out_f = f"{SCRATCH}/Xron_Project/Benchmark/HEK293T/g2t_dict.pkl"
    print("Reading the t2g dictionary...")
    if not os.path.exists(t2g_f):
        print("t2g dictionary not found. Please run the first test case first.")
        exit(1)
    with open(t2g_f, "rb") as f:
        t2g_dict = pickle.load(f)
    print("Creating g2t dictionary...")
    g2t_dict = reverse_mapping(t2g_dict)
    print("Saving g2t dictionary...")
    with open(out_f, "wb+") as f:
        pickle.dump(g2t_dict, f)
    
# %%
