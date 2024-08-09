# Distill training data from basecalled alignments
# Immunoprecipitation result can be used ground truth training data
import os
import lmdb
import pickle
import pandas as pd
import pysam
import pod5 as p5
import numpy as np
import multiprocessing as mp
import queue as Queue
import time
from tqdm import tqdm
from functools import lru_cache
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

def get_partition(total_len, partition_len):
    "get the partition segment"
    "for segment < half partition_len, merge it to the previous segment"
    "for segment > half partition_len, split it to two segments"
    if total_len < partition_len:
        return [total_len]
    else:
        partitions = []
        for i in range(partition_len,total_len,partition_len):
            partitions.append(i)
        if total_len - partitions[-1] < partition_len//2:
            partitions[-1] = total_len
        else:
            partitions.append(total_len)
        return partitions
    

class Balancer(object):
    def __init__(self, config):
        self.ref_counts = defaultdict(int)
        self.partition_dict = {}
        self.partition_over = config.get('partition_over',100000) #Partition the reference for length over this value
        self.max_single_ref = config.get('max_single_ref',3) #Maximum ratio of reads from a single reference to the median of all references
        self.mean = 1e-6
        self.std = 1

    def load_ref(self, ref_f):
        self.ref = pysam.FastaFile(ref_f)
        for ref in self.ref.references:
            ref_len = self.ref.get_reference_length(ref)
            partitions = get_partition(ref_len, self.partition_over)
            self.partition_dict[ref] = np.asarray(partitions)

    def get_statistic(self):
        ref_counts = np.asarray(list(self.ref_counts.values()))
        if len(ref_counts) == 0:
            return 0,1
        return np.mean(ref_counts), np.std(ref_counts)
    
    def update_statistic(self):
        self.mean, self.std = self.get_statistic()

    def add(self, ref_id, start, end):
        partition_idxs = self.get_partition_idx(ref_id,start,end)
        for idx in partition_idxs:
            self.ref_counts[f"{ref_id}@{idx}"] += 1
        self.update_statistic()
    
    def balance_check(self, ref_id, start, end):
        partition_idxs = self.get_partition_idx(ref_id,start,end)
        for idx in partition_idxs:
            if self.ref_counts[f"{ref_id}@{idx}"] > self.mean + self.max_single_ref * max(self.std,1):
                return False
        self.add(ref_id,start,end)
        return True
    
    @lru_cache(maxsize = 10)
    def get_partition_idx(self,ref_id,start,end):
        partition_table = self.partition_dict[ref_id]
        start_id = np.where(partition_table>=start)[0][0]
        end_id = np.where(partition_table>=end)[0][0]
        idxs = np.arange(start_id,end_id+1)
        return idxs

class Distiller(object):
    def __init__(self, config, result_queue, filters = [], ):
        self.config = config
        self.balancer = Balancer(config)
        self.reverse = True if "RNA" in config['kit'] else False
        self.modification_code = config['modification_code']
        self.chunk_len = config['chunk_len']
        self.run_status = defaultdict(int)
        self.filter = and_filters(filters)
        self.pod5 = None
        self.queue = result_queue

    def load_data(self,bam_f, pod5_f, ref_f, immun_f = None,batch_ids = None):
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
        if batch_ids:
            def batch_iter():
                file = p5.Reader(pod5_f)
                for batch in file.read_batches(batch_selection = batch_ids, preload = {"samples"}):
                    for read in batch.reads():
                        yield read
            self.pod5_iter = batch_iter()
        else:
            self.pod5_iter = p5.DatasetReader(pod5_f).reads(preload = {"samples"})
        self.balancer.load_ref(ref_f)
        if self.immun_f:
            self.immun = pd.read_csv(self.immun_f)
        else:
            self.immun = None

    def run(self,rank = 0):
        #select signal and sequences given the indexer mapping
        #iterate over the read ids of pod5 file 
        chunks, seqs = [], []
        if rank == 0:
            t = tqdm(self.pod5_iter, mininterval=0.1)
        else:
            t = self.pod5_iter
        for pod5_read in t:
            read_id = str(pod5_read.read_id)
            try:
                index_map = self.indexer[read_id]
            except KeyError:
                self.run_status["Unaligned"] += 1
                continue
            ref_id = index_map['reference_name']
            for read in self.align.fetch(ref_id):
                if read.query_name == read_id:
                    break
            if read.query_name != read_id:
                self.run_status["Read not found in BAM"] += 1
                continue
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
                if not self.balancer.balance_check(ref_id, ref_s, ref_e):
                    self.run_status["Balance out"] += 1
                    i = seq_end
                    continue

                #If immunoprecipitation result is provided, modify the sequence at the modification site
                if self.immun is not None:
                    seq = self.seq_mod(seq, list(range(ref_s,ref_e)), ref_id, self.immun)

                #Filter the sequence
                record = {"signal":np.pad(sig_chunk,(0,self.chunk_len-len(sig_chunk))),
                        "seq":seq,
                        "seq_len":len(seq),
                        "signal_len":len(sig_chunk),
                        "read_id":read_id,
                        "ref_id":ref_id,
                        "ref_start":ref_s,
                        "ref_end":ref_e}
                if self.filter(record):
                    #check if queue is full
                    while self.queue.full():
                        pass
                    self.queue.put(record)
                    self.balancer.add(ref_id, ref_s, ref_e)
                    i = seq_end
                    self.run_status["Processed"] += 1
                else:
                    self.run_status["Filtered"] += 1
                    i += 5
            if rank == 0:
                t.set_postfix(self.run_status)
            if len(chunks) > self.config['n_max']:
                break


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
    rev_rate = record['signal_len'] / len(record['seq'])
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
    out_f = f"{base}/extracted_lmdb/"
    os.makedirs(out_f, exist_ok =True)
    runners = 7
    n_batch_per_runner = None #each batch 1000 files

    #optional Immunoprecipitation file
    config = {"index_backend":"lmdb",
              "kit": "SQK-RNA004",
              "chunk_len":2000,
              "modification_code":'A+a',
              "n_max":1000000,
              "partition_over":100000,
              'max_single_ref':2,
              }
    
    def worker(batch_ids,process_id,result_queue):
        distiller = Distiller(config = config,
            filters = [seq_filter, mono_filter, bps_rate_filter,seq_sanity_check],
            result_queue = result_queue)
        distiller.load_data(bam_f,
                            pod5_f = pod5_f,
                            ref_f = ref_f,
                            immun_f = immun_f,
                            batch_ids = batch_ids)
        distiller.run(process_id)

    def write_queue_to_lmdb(queue, lmdb_path, map_size=1e9, timeout=1, max_trail=5):
        """
        Write the contents of a queue to an LMDB dataset, waiting for new items if the queue is empty.

        Parameters:
        - queue: A queue containing dictionaries with keys "signal", "seq", "seq_len", and "signal_len".
        - lmdb_path: Path to the LMDB file.
        - map_size: Maximum size database may grow to; used to size the memory mapping. Defaults to 1GB.
        - timeout: Time to wait for new items if the queue is empty (in seconds). Defaults to 1 seconds.
        - max_trail: Maximum number of trails of attempting acquire data from queue before exiting. Defaults to 5 trails.
        """
        env = lmdb.open(lmdb_path, map_size=int(map_size))
        idx,trails = 0,0
        txn = env.begin(write=True)
        while True:
            try:
                item = queue.get(timeout=timeout)
                if not all(key in item for key in ["signal", "seq", "seq_len", "signal_len"]):
                    raise ValueError("Dictionary missing required keys: 'signal', 'seq', 'seq_len', 'signal_len'")

                # Serialize the item
                serialized_item = pickle.dumps(item)
                # write to lmdb
                success = False
                while not success:
                    try:
                        txn.put(str(idx).encode(), serialized_item)
                        success = True
                        idx += 1
                    except lmdb.MapFullError:
                        txn.abort()
                        # double the map_size
                        curr_limit = env.info()['map_size']
                        new_limit = curr_limit*2
                        env.set_mapsize(new_limit) # double it
                        txn = env.begin(write=True)
                if idx % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                trails = 0
            except Queue.Empty:
                if trails > max_trail:
                    break
                time.sleep(timeout)  # Wait for a while before checking again
                trails += 1
        txn.commit()
        env.close()
    
    file = p5.Reader(pod5_f)
    batches = list(range(file.batch_count))
    start_idx = 0
    processes_idx = 0
    processes = []
    result_queue = mp.Queue()
    if n_batch_per_runner is None:
        n_batch_per_runner = len(batches) // runners
    while start_idx < len(batches):
        batch_ids = batches[start_idx:start_idx+n_batch_per_runner]
        start_idx += len(batch_ids)
        p = mp.Process(target = worker, args = (batch_ids,processes_idx,result_queue))
        processes_idx += 1
        p.start()
        processes.append(p)

    #Start the queue consumer, write the data to the file
    lmdb_out = f"{out_f}/"
    p = mp.Process(target = write_queue_to_lmdb, args = (result_queue, lmdb_out))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()
    