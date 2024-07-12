# Build an index dictionary for sequence position -> reference position & signal position
import re
import pysam
import pod5 
import pickle
import zlib
import lmdb
from collections import defaultdict
from tqdm import tqdm
import numpy as np
DATA_TYPE = np.int32
START_H_PATTERN = re.compile(r'^(\d+)H')
END_H_PATTERN = re.compile(r'.(\d+)H$')

def get_hc(read):
    cigar_string = read.cigarstring
    # Find all occurrences of a number followed by 'H' at the start or end
    start_match = START_H_PATTERN.match(cigar_string)
    end_match = END_H_PATTERN.search(cigar_string)
    
    start_h = int(start_match.group(1)) if start_match else 0
    end_h = int(end_match.group(1)) if end_match else 0
    
    return start_h, end_h

def find_bases(move_table):
    """Find an index mapping for the sequence position to the signal position
    from the move table
    Args:
        move_table: np.array, the move table from the mv:b:c tag, e.g. [1,0,0,1,1,0,0,1,...]
    Returns:
        seq_idx: np.array, the sequence index
        sig_pos: list of tuple, the start and end of the signal position of the same sequence index
    """
    seq_idx = np.cumsum(move_table,dtype = DATA_TYPE)
    changes = np.diff(seq_idx, prepend=1)
    indices = np.where(changes != 0)[0]
    
    starts = indices
    ends = np.append(indices[1:], len(move_table))
    
    sig_pos = list(zip(starts, ends))
    return np.unique(seq_idx)-1, sig_pos

class Indexer(dict):
    def __init__(self, compress=False):
        self.run_stat = defaultdict(int)
        self.compress = compress
        self.db = None

    def load(self,bam_f):
        index_f = bam_f + ".index"
        env = lmdb.open(index_f, readonly=True)
        self.db = env.begin()
    
    def __getitem__(self,key):
        if self.db is None:
            raise ValueError("No database has been loaded!")
        return pickle.loads(zlib.decompress(self.db[key]))
    
    def _write_to_lmdb(self, db,key,value):
        """
        Write (key,value) to db
        """
        success = False
        while not success:
            txn = db.begin(write=True)
            try:
                txn.put(key, value)
                txn.commit()
                success = True
            except lmdb.MapFullError:
                txn.abort()

                # double the map_size
                curr_limit = db.info()['map_size']
                new_limit = curr_limit*2
                db.set_mapsize(new_limit) # double it

    def build(self, bam_f):
        sam_f = pysam.AlignmentFile(bam_f, "rb")
        out_f = bam_f + ".index"
        env = lmdb.open(out_f, map_size=int(1e11))  # Adjust map_size as needed
        with tqdm(sam_f, mininterval=0.1) as t:
            for read in t:
                if read.is_unmapped:
                    self.run_stat['Unaligned'] += 1
                    continue        
                if read.is_secondary:
                    self.run_stat['Secondary alignment'] += 1
                    continue
                ref_contig = read.reference_name
                qr_map = read.get_aligned_pairs(matches_only=False, with_seq=False)
                hc_start, hc_end = get_hc(read)
                read_id = read.query_name
                try:
                    trim_length = read.get_tag("ts:i")
                    move_table = read.get_tag("mv:B")
                except KeyError:
                    self.run_stat['No moving table'] += 1
                    if self.run_stat['No moving table'] // (self.run_stat['Success'] + 1) > 100:
                        raise ValueError("No moving table in the basecalled bam file, is --emit-moves tag used during basecall?")
                    continue
                # Get signal position by accumulating the move_table
                stride = move_table[0]
                seq_idx, sig_pos = find_bases(move_table[1:])
                curr = {
                    'seq2sig': {i: s for i, s in zip(seq_idx, sig_pos)},
                    'sig2seq': [s[0] * stride + trim_length for s in sig_pos],
                    'stride': stride,
                    'trim_length': trim_length,
                    'reference_name': ref_contig
                }
                if read.is_reverse:
                    hc_start, hc_end = hc_end, hc_start
                    q_transform = lambda x: len(read.seq) - x - 1 if x is not None else None
                else:
                    q_transform = lambda x: x
                curr['query_reference_map'] = {q_transform(x[0]): x[1] for x in qr_map}
                curr['hard_clip'] = (hc_start, hc_end)
                serailzed = pickle.dumps(curr)
                if self.compress:
                    serailzed = zlib.compress(serailzed)
                self._write_to_lmdb(env, read_id.encode('utf-8'), serailzed)
                self.run_stat['Success'] += 1

                # Early stop for testing
                # if self.run_stat['Success'] >= 10000:
                #     break

                # Update run status
                t.set_postfix(self.run_stat, refresh=False)

def args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", type=str, required=True, help="The aligned bam file")
    parser.add_argument("--compress", action="store_true", help="Compress the index entry")
    return parser.parse_args()

if __name__ == "__main__":
    import time
    start = time.time()
    args = args()
    input_bam = args.bam
    runner = Indexer(compress=args.compress)
    runner.build(input_bam)
    print(f"Time elapsed: {time.time() - start}")
