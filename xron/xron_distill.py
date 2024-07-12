# Distill training data from basecalled alignments
# Immunoprecipitation result can be used ground truth training data
import os
import pandas as pd
import pysam
import time
import pod5 as p5
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from xron.utils.fastIO import Indexer
from xron.utils.seq_op import norm_by_noisiest_section

def and_filters(filters):
    def and_filter(kwargs):
        for f in filters:
            if not f(kwargs):
                return False
        return True
    return and_filter

class Balancer(object):
    #TODO impelement a balancer to balance the 
    #number of reads selected from each reference
    pass


class Distiller(object):
    def __init__(self, config, filters = []):
        self.config = config
        self.reverse = True if "RNA" in config['kit'] else False
        self.modification_code = config['modification_code']
        self.chunk_len = config['chunk_len']
        self.run_status = defaultdict(int)
        self.filter = and_filters(filters)
        self.pod5 = None

        #debugging code
        self.accumulate_time = 0
    
    def load_data(self,bam_f, pod5_f, ref_f, immun_f = None):
        self.bam_f = bam_f
        self.index_f = index_f
        self.ref_f = ref_f
        self.immun_f = immun_f
        self.pod5_f = pod5_f
        index_backend = self.config.get('index_backend','lmdb')
        self.indexer = Indexer(backend = index_backend)
        self.indexer.load(self.index_f)
        self.align = pysam.AlignmentFile(self.bam_f, "rb")
        self.ref = pysam.FastaFile(self.ref_f)
        self.pod5 = p5.DatasetReader(pod5_f)
        if self.immun_f:
            self.immun = pd.read_csv(self.immun_f)
        else:
            self.immun = None

    def run(self,out_f):
        #select signal and sequences given the indexer mapping
        #iterate over the read ids of pod5 file 
        chunks, seqs = [], []
        with tqdm(self.pod5.reads(), mininterval=0.1) as t:
          for pod5_read in t:
            read_id = str(pod5_read.read_id)
            try:
                index_map = self.indexer[read_id]
            except KeyError:
                self.run_status["Unaligned"] += 1
                continue
            ref_id = index_map['reference_name']
            start = time.time()
            for read in self.align.fetch(ref_id):
                if read.query_name == read_id:
                    break
            if read.query_name != read_id:
                self.run_status["Read not found in BAM"] += 1
                continue
            self.accumulate_time += time.time() - start
            seq = read.query_sequence
            ref = self.ref.fetch(read.reference_name)
            if read.is_reverse:
                self.run_status['Skip reverse'] += 1
                continue
            if seq is None:
                self.run_status["No seq in SAM"] += 1
                continue
            
            if ref_id != read.reference_name:
                self.run_status["Reference name mismatch"] += 1
                continue
            try:
                signal = self.extract_signal(pod5_read)
            except RuntimeError:
                self.run_status["Signal missing in pod5"] += 1
                continue
            qr_map = index_map['query_reference_map']
            hc_start, hc_end = index_map['hard_clip']
            seq2sig = self.get_sig_map(index_map, len(signal))
            sig_ends = self.get_sig_milestones(index_map, len(signal)) #get the end positions of each base
            i = hc_start
            while (i+1)<(len(seq)-hc_end):
                sig_s,_ = seq2sig[i]

                #Find the closest seq idx for sig + 4000
                sig_e = sig_s+self.chunk_len
                try:
                    seq_end = np.where(sig_ends<=sig_e)[0][-1]
                except:
                    i += 1
                    continue
                if (i-hc_start not in qr_map or qr_map[i-hc_start] is None):
                    i += 1
                    continue
                while (seq_end - hc_start not in qr_map) or (qr_map[seq_end-hc_start] is None):
                    seq_end -= 1
                    if seq_end <= i:
                        break
                if seq_end <= i:
                    self.run_status["Out of qr mapping boundary"] += 1
                    i += 1
                    continue
                #Get the signal chunk according to the sequence idxs
                sig_e = seq2sig[seq_end][1]
                sig_chunk = signal[sig_s:sig_e]

                #Get the refernece sequence
                ref_s,ref_e = qr_map[i-hc_start],qr_map[seq_end-hc_start]+1
                seq = ref[ref_s:ref_e]
                if len(seq) != ref_e - ref_s:
                    self.run_status["Reference seq length mismatch"] += 1
                    i += 1
                    continue

                #If immunoprecipitation result is provided, modify the sequence at the modification site
                if self.immun is not None:
                    seq = self.seq_mod(seq, list(range(ref_s,ref_e)), ref_id, self.immun)

                #Filter the sequence
                if self.filter({'seq':seq,'signal':sig_chunk}):
                    #write to the out file
                    seqs.append(seq)
                    chunks.append(sig_chunk)
                    i = seq_end
                    self.run_status["Processed"] += 1
                else:
                    self.run_status["Filtered"] += 1
                    i += 5
            t.set_postfix(self.run_status)
            if len(chunks) > self.config['n_max']:
                break
        #write to the out file
        chunk_lens = np.asarray([len(c) for c in chunks])
        #pad chunk to chunk_len
        chunks = np.array([np.pad(c,(0,self.chunk_len-len(c))) for c in chunks])
        seq_lens = np.asarray([len(s) for s in seqs])
        seqs = np.asarray(seqs)
        np.save(os.path.join(out_f,"chunks.npy"),chunks)
        np.save(os.path.join(out_f,"seqs.npy"),seqs)
        np.save(os.path.join(out_f,"chunk_lens.npy"),chunk_lens)
        np.save(os.path.join(out_f,"seq_lens.npy"),seq_lens)

        #Time testing code
        print(f"Time elapsed: {self.accumulate_time}")
    

    def extract_signal(self, read):
        signal,_,_ = norm_by_noisiest_section(read.signal_pa)
        signal = signal[::-1] if self.reverse else signal
        return signal

    def fetch_signal(self, id):
        if self.pod5 is None:
            raise ValueError("No pod5 file has been loaded")
        read = next(self.pod5.reads(selection =[id]))
        return self.extract_signal(read)

    def fetch_signals(self, ids):
        if self.pod5 is None:
            raise ValueError("No pod5 file has been loaded")
        signals = []
        for read in self.pod5.reads(selection = ids):
            # Get the signal data and sample rate
            signals.append(self.extract_signal(read))
        return signals

    def seq_mod(self, 
                seq, 
                ref_idxs, 
                ref_name,
                immun_df, 
                label_column = "modification_status",
                ref_name_column = "transcript_id",
                ref_pos_column = "transcript_pos"):
        """
        Annotate the modified sequence given the query reference map and immunoprecipitation result
        Args:
            seq: str, the (reference) sequence to be corrected
            ref_idxs: list, the position of the sequence in the reference, should have same length as seq
            ref_name: str, the reference name
            immun_df: pd.DataFrame, the immunoprecipitation result
            label_column: str, the column name of the modification status in the immun_dif
            ref_name_column: str, the column name of the reference name in the immun_dif
            ref_pos_column: str, the column name of the reference position in the immun_dif
        """
        assert len(seq) == len(ref_idxs), "The length of the sequence and reference position should be the same"
        entries = immun_df[immun_df[ref_name_column] == ref_name]
        can_base = self.modification_code.split("+")[0]
        if entries.empty:
            return seq
        seq = list(seq)
        for i, (ref_pos,b) in enumerate(zip(ref_idxs,seq)):
            entry = entries[entries[ref_pos_column] == ref_pos]
            if entry.empty:
                continue
            if entry[label_column].values[0] and (b ==can_base):
                seq[i] = "M"
        return "".join(seq)

    def _sig_pos(self, index_map, sig_idx, sig_len):
        """
        Given the index_map and signal index, return the signal position
        by caculating the sig_idx * stride + trim_length
        For reverse signal, the signal position will also be reversed, e.g. sig_pos = sig_len - sig_pos - 1
        """
        stride = index_map['stride']
        trim = index_map['trim_length']
        sig_idx = sig_idx * stride + trim
        if self.reverse:
            sig_idx = sig_len - sig_idx - 1
        return sig_idx
    
    def get_sig_map(self, index_map,sig_len):
        """
        Given the index_map and signal length, return the signal map
        """
        if self.reverse:
            n_seq = max(index_map['seq2sig'].keys())
            return {n_seq - i: (self._sig_pos(index_map,e,sig_len),self._sig_pos(index_map,s,sig_len)) for i,(s,e) in index_map['seq2sig'].items()}
        else:
            return {i: (self._sig_pos(index_map,s,sig_len),self._sig_pos(index_map,e,sig_len)) for i,(s,e) in index_map['seq2sig'].items()}

    def get_sig_milestones(self, index_map, sig_len):
        """
        Given the index_map and signal length, return the signal milestones
        """
        if self.reverse:
            return sig_len - np.asarray(index_map['sig2seq'])[::-1]
        else:
            sorting = np.argsort(index_map['seq2sig'].keys())
            return np.asarray([p[1] for p in index_map['seq2sig'].values()])[sorting]

    def __del__(self):
        self.align.close()
        self.ref.close()
        self.indexer.__del__()

def seq_filter(record, min_seq_len = 3):
    if len(record['seq']) < min_seq_len:
        return False
    return True

def mono_filter(record, max_mono_proportion = 0.8):
    """Filter out sequence if propotion of one single base is greater than max_mono_proportion"""
    seq = record['seq']
    mono_prop = max([seq.count(b)/len(seq) for b in set(seq)])
    if mono_prop > max_mono_proportion:
        return False
    return True

def bps_rate_filter(record, 
                    bps_rate = 130, 
                    max_bps_rate = 2., 
                    min_bps_rate = 0.5, 
                    sampling_rate = 4096):
    """Filter out sequence if the moving rate is greater than max_moving_rate"""
    min_rev_rate = sampling_rate / (max_bps_rate * bps_rate)
    max_rev_rate = sampling_rate / (min_bps_rate * bps_rate)
    rev_rate = len(record['signal']) / len(record['seq'])
    if rev_rate > max_rev_rate or (rev_rate < min_rev_rate):
        return False
    return True

def seq_sanity_check(record):
    """Check if the sequence is valid"""
    seq = record['seq']
    bases = set(seq)
    if not bases.issubset(set("ACGTM")):
        return False
    return True


if __name__ == "__main__":
    base = "/data/HEK293T_RNA004/"
    bam_f = f"{base}/aligned.sorted.bam"
    pod5_f = f"{base}/SGNex_Hek293T_directRNA_replicate5_run1.pod5"
    index_f = f"{bam_f}.index"
    ref_f = f"{base}/Homo_sapiens.GRCh38.cdna.ncrna.fa"
    immun_f = f"{base}/m6A_site_m6Anet_DRACH_HEK293T_with_transcripts.csv"
    out_f = f"{base}/extracted/"
    os.makedirs(out_f, exist_ok =True)

    #optional Immunoprecipitation file
    config = {"index_backend":"lmdb",
              "kit": "SQK-RNA004",
              "chunk_len":2000,
              "modification_code":'A+a',
              "n_max":1000000, }
    
    distiller = Distiller(config = config,
        filters = [seq_filter, mono_filter, bps_rate_filter,seq_sanity_check])
    distiller.load_data(bam_f,
                        pod5_f = pod5_f,
                        ref_f = ref_f,
                        immun_f = immun_f)
    distiller.run(out_f)