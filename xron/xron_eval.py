"""
Created on Wed Sep 15 08:44:24 2021

@author: Haotian Teng
"""
import os
import sys
import h5py
import torch
import shutil
import argparse
import numpy as np
from datetime import datetime
from functools import partial
from tqdm import tqdm
from xron.xron_model import CRNN, CONFIG
from xron.xron_train_base import load_config
from xron.utils.seq_op import raw2seq,fast5_iter,norm_by_noisiest_section,list2string
from xron.utils.easy_assembler import simple_assembly_qs
from itertools import groupby 

##Debug module
from timeit import default_timer as timer
##

def load(model_folder,net,device = 'cuda'):
    ckpt_file = os.path.join(model_folder,'checkpoint')
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
    ckpt = torch.load(os.path.join(model_folder,latest_ckpt),
                      map_location=device)
    if isinstance(ckpt,dict):
        net.load_state_dict(ckpt["encoder"])
    else:
        net.load_state_dict(ckpt)
    net.to(device)

def chunk_feeder(fast5_f,config):
    iterator = fast5_iter(fast5_f)
    e = config.EVAL
    batch_size,chunk_len,device,offset = e['batch_size'],e['chunk_len'],e['device'],e['offset']
    chunks,meta_info = [],[]
    for read_h,signal,fast5_f,read_id in tqdm(iterator):
        read_len = len(signal)
        signal = norm_by_noisiest_section(signal,offset = offset)[0].astype(np.float16)
        if config.CTC['mode'] == "rna":
            signal = signal[::-1]
        current_chunks = np.split(signal,np.arange(0,read_len,chunk_len))[1:]
        last_chunk = current_chunks[-1]
        last_chunk_len = len(last_chunk)
        current_chunks[-1]= np.pad(last_chunk,(0,args.chunk_len-last_chunk_len),'constant',constant_values = (0,0))
        chunks += current_chunks
        meta_info += [(fast5_f,read_id,str(args.chunk_len))]*(len(current_chunks)-1) 
        meta_info += [(fast5_f,read_id,last_chunk_len)]
        while len(chunks) >= batch_size:
            curr_chunks = chunks[:batch_size]
            curr_meta = meta_info[:batch_size]
            yield torch.from_numpy(np.stack(curr_chunks,axis = 0)[:,None,:].astype(np.float32)).to(device),curr_meta
            chunks = chunks[batch_size:]
            meta_info = meta_info[batch_size:]
    if len(chunks):
        chunks += [chunks[-1]]*(batch_size - len(chunks))
        meta_info += [(None,None,None)]*(batch_size - len(chunks))
        yield torch.from_numpy(np.stack(chunks,axis = 0)[:,None,:].astype(np.float32)).to(device),curr_meta

def qs(consensus, consensus_qs, output_standard='phred+33'):
    """Calculate the quality score for the consensus read.

    Args:
        consensus (Int): 2D Matrix (bases, read length) given the count of base on each position.
        consensus_qs (Float): 1D Vector given the mean of the difference between the highest logit and second highest logit.
        output_standard (str, optional): Defaults to 'phred+33'. Quality score output format.

    Returns:
        quality score: Return the queality score as int or string depending on the format.
    """

    sort_ind = np.argsort(consensus, axis=0)
    L = consensus.shape[1]
    sorted_consensus = consensus[sort_ind, np.arange(L)[np.newaxis, :]]
    sorted_consensus_qs = consensus_qs[sort_ind, np.arange(L)[np.newaxis, :]]
    quality_score = 10 * (np.log10((sorted_consensus[-1, :] + 1) / (
        sorted_consensus[-2, :] + 1))) + sorted_consensus_qs[-1, :] / sorted_consensus[-1, :] / np.log(10)
    if output_standard == 'number':
        return quality_score.astype(int)
    elif output_standard == 'phred+33':
        q_string = [chr(x + 33) for x in quality_score.astype(int)]
        return ''.join(q_string)

def vaterbi_decode(logits:torch.tensor):
    """
    Vaterbi decdoing algorithm

    Parameters
    ----------
    logits : torch.tensor
        Shape L-N-C

    Returns
    -------
        sequence: A length N list contains ndarray arrays.
        moves: A length N list contains the moves array.

    """
    sequence = np.argmax(logits,axis = 2)
    sequence = sequence.T #L,N -> N,L
    sequence,moves = raw2seq(sequence)
    return sequence,list(moves.astype(np.int))

def consensus_call(seqs,metas,moves,assembly_func,logits = None):
    ids,counts = np.unique([x[1] for x in metas],return_counts = True)
    curr = [(x,y,z,w) for x,y,z,w in zip(seqs,metas,logits,moves) if y[1] == metas[0][1]]
    curr_seqs,curr_meta,qs_list,moves = zip(*curr)
    css = assembly_func(curr_seqs,qs_list = qs_list,jump_step_ratio = 0.9)
    curr_move = np.asarray(moves).flatten()
    fast5f = set([x[0] for x in curr_meta])
    id = set([x[1] for x in curr_meta])
    assert len(fast5f)==1 and len(id)==1
    return css,id.pop(),fast5f.pop(),curr_move

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

class Writer(object):
    def __init__(self,write_dest,config):
        self.css_seqs = []
        self.read_ids = []
        self.fast5_fs = []
        self.qs = []
        self.moves = []
        self.count = 0
        self.base_dict = {i:b for i,b, in enumerate(config.CTC['alphabeta'])}
        self.prefix = write_dest
        self.prefix_fastq = os.path.join(self.prefix,'fastqs')
        os.makedirs(self.prefix_fastq,exist_ok = True)
        self.prefix_fast5 = os.path.join(self.prefix,'fast5s')
        self.seq_batch = config.EVAL['seq_batch']
        self.format = config.EVAL['format']
        if self.format == "fast5":
            os.makedirs(self.prefix_fast5,exist_ok = True)
        self.stride = config.CNN['Layers'][-1]['stride']
        
    def add(self,css,read_id,fast5f,move):
        if(len(css)) == 2:
            self.css_seqs.append(list2string(np.argmax(css[0],axis=0),base_type = self.base_dict))
            self.qs.append(qs(css[0],css[1]))
        else:
            self.css_seqs.append(list2string(np.argmax(css,axis = 0),base_type = self.base_dict))
        self.moves.append(move)
        self.read_ids.append(read_id)
        self.fast5_fs.append(fast5f)
        if len(self.css_seqs) > self.seq_batch:
            self.flush()
    
    def flush(self):
        if self.format == "fastq":
            self.write_fastq("%s/%d.fastq"%(self.prefix_fastq,self.count))
        if self.format == "fast5":
            self.write_fastq("%s/%d.fastq"%(self.prefix_fastq,self.count))
            self.write_fast5(self.prefix_fast5)
        self.count += 1
        self.css_seqs = []
        self.read_ids = []
        self.fast5_fs = []
        self.moves = []
        self.qs = []
        
    def write_fasta(self,output_f):
        with open(output_f,mode = 'w+') as f:
            for seq,name in zip(self.css_seqs,self.read_ids):
                f.write(">%s\n%s\n"%(name,seq))
    
    def write_fast5(self,output_f):
        for seq,name,fast5,q,move in zip(self.css_seqs,self.read_ids,self.fast5_fs,self.qs,self.moves):
            dest = os.path.join(output_f,os.path.basename(fast5))
            if not os.path.isfile(dest):
                shutil.copy(fast5,dest)
            with h5py.File(dest,mode='a') as root:
                read_h = root['read_%s/Analyses'%(name)]
                existed_basecall = [x for x in list(read_h.keys()) if 'Basecall' in x]
                curr_group = read_h.create_group('Basecall_1D_%03d'%(len(existed_basecall)))
                curr_group.attrs.create('name','Xron')
                curr_group.attrs.create('time_stamp','%s'%(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))
                result_h = curr_group.create_group('BaseCalled_template')
                result_h.create_dataset("Fastq",data = "@%s\n%s\n+\n%s\n"%(name,seq,q))
                result_h.create_dataset("Move",data = move,dtype='i8')
                summary_h = curr_group.create_group("Summary")
                summary_h = summary_h.create_group("basecall_1d_template")
                summary_h.attrs.create('block_stride',self.stride)
    
    def write_fastq(self,output_f):
        with open(output_f,mode = 'w+') as f:
            for seq,name,q in zip(self.css_seqs,self.read_ids,self.qs):
                f.write("@%s\n%s\n+\n%s\n"%(name,seq,q))
    
def main(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    class CTC_CONFIG(CONFIG):
        CTC = {"beam_size":args.beam,
               "beam_cut_threshold":0.05,
               "alphabeta": "ACGTM",
               "mode":"rna"}
    class CALL_CONFIG(CTC_CONFIG):
        EVAL = {"batch_size":args.batch_size,
                "chunk_len":args.chunk_len,
                'device':args.device,
                'assembly_method':args.assembly_method,
                'seq_batch':4000,
                'format':'fast5' if args.fast5 else 'fastq',
                'offset':args.offset}
    config = CALL_CONFIG()
    print("Construct and load the model.")
    model_f = args.model_folder
    config_old = load_config(os.path.join(model_f,"config.toml"))
    config_old.EVAL = config.EVAL #Overwrite training config.
    config = config_old
    if args.config:
        config = load_config(args.config)
    net = CRNN(config)
    load(args.model_folder,net,device = args.device)
    print("Begin basecall.")
    net.eval()
    df = chunk_feeder(args.input, config)
    seqs,metas,moves = ([],[],[])
    writer = Writer(args.output,config)
    assembly_func = partial(simple_assembly_qs,
                            kernal = args.assembly_method,
                            alphabeta = config.CTC["alphabeta"])
    nn_time = 0.
    assembly_time = 0.
    writing_time = 0.
    with torch.no_grad():
        for chunk,meta in df:
            start = timer()
            logits = net(chunk).cpu().numpy() #Length-Batch-Channel
            end = timer()
            nn_time+= (end-start)
            start = timer()
            qs_list = np.max(logits,axis = 2).T
            sequence,move = vaterbi_decode(logits)
            seqs += sequence
            metas += meta
            moves += move
            if (None,None,None) in metas:
                print("Remove the padding chunks.")
                remain = [[x,y,z] for x,y,z in zip(seqs,metas,moves) if y[1] != None]
                seqs,metas,moves = zip(*remain)
                seqs,metas,moves = list(seqs),list(metas),list(moves)
            while not all_equal(x[1] for x in metas):
                r = consensus_call(seqs,metas,moves,assembly_func,logits = qs_list)
                start_w = timer()
                writer.add(*r)
                end_w = timer()
                writing_time += (end_w-start_w)
                remain = [[x,y,z] for x,y,z in zip(seqs,metas,moves) if y[1] != metas[0][1]]
                seqs,metas,moves = zip(*remain)
                seqs,metas,moves = list(seqs),list(metas),list(moves)
            end = timer()
            assembly_time += (end-start)
    start_w = timer()
    writer.flush()
    end_w = timer()
    writing_time += (end_w-start_w)
    print("NN_time:%f,assembly_time:%f,writing_time:%f"%(nn_time,assembly_time,writing_time))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Call a xron model on dataset.')
    parser.add_argument('-i', '--input', required = True,
                        help = "The input folder contains the fast5 files.")
    parser.add_argument('-m', '--model_folder', required = True,
                        help = "The folder contains the model.")
    parser.add_argument('-o', '--output', required = True,
                        help = "The output folder.")
    parser.add_argument('--fast5',action = "store_true",dest = "fast5",
                        help = "If output fast5 files.")
    parser.add_argument('--device', default = 'cuda',
                        help="The device used for training, can be cpu or cuda.")
    parser.add_argument('--batch_size', default = None, type = int,
                        help="Training batch size, default use the maximum size of the GPU memory, batch size and memory consume relationship: 200:6.4GB, 400:11GB, 800:20GB, 1200:30GB.")
    parser.add_argument('--chunk_len', default = 2000, type = int,
                        help="The length of each chunk signal.")
    parser.add_argument('--beam',default = 1,type = int,
                        help="The width of CTC beam search decoder.")
    parser.add_argument('--config', default = None,
                        help = "Training configuration.")
    parser.add_argument('--threads', type = int, default = None,
                        help = "Number of threads used by Pytorch")
    parser.add_argument('--assembly_method',type = str, default = "stick",
                        help = "Assembly method used, can be glue, global, simple and stick.")
    parser.add_argument('--offset',type = float,default = 0.0,
                        help = "Manual set a offset to the normalized signal.")
    args = parser.parse_args(sys.argv[1:])
    MEMORY_PER_BATCH_PER_SIGNAL=13430. #KB
    t = torch.cuda.get_device_properties(0).total_memory
    if not args.batch_size:
        args.batch_size = int(t/MEMORY_PER_BATCH_PER_SIGNAL/args.chunk_len//100*100)
        print("Auto configure to use %d batch_size for a total of %.1f GB memory."%(args.batch_size,t/1024**3))
    os.makedirs(args.output,exist_ok=True)
    main(args)
