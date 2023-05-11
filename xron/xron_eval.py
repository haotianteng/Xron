import os
import sys
import h5py
import toml
import torch
import shutil
import inspect
import argparse
import numpy as np
from typing import Callable
from datetime import datetime
from itertools import groupby 
from functools import partial
from fast_ctc_decode import beam_search
from boostnano.boostnano_model import CSM
from boostnano.boostnano_eval import evaluator
from xron.xron_model import CRNN, CONFIG
from xron.xron_train_base import load_config
from xron.utils.seq_op import raw2seq,fast5_iter,norm_by_noisiest_section,med_normalization,list2string
from xron.utils.easy_assembler import simple_assembly_qs


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

def chunk_feeder(fast5_f,config,boostnano_evaluator = None):
    iterator = fast5_iter(fast5_f,tqdm_bar = True)
    e = config.EVAL
    batch_size,chunk_len,device,offset,overlay = e['batch_size'],e['chunk_len'],e['device'],e['offset'],e['overlay']
    chunks,meta_info = [],[]
    for read_h,signal,fast5_f,read_id in iterator:
        read_len = len(signal)
        if e['mode'] == 'rna' or e['mode'] == 'rna-meth':
            if boostnano_evaluator is None:
                signal = signal[::-1]
                signal = norm_by_noisiest_section(signal,offset = offset)[0].astype(np.float32)
            else:
                (decoded,path,locs) = boostnano_evaluator.eval_sig(signal,1000)
                locs = read_len - locs
                signal = signal[::-1][:locs[0]]
                if len(signal) == 0:
                    continue
                signal = med_normalization(signal).astype(np.float32)
                read_len = len(signal)
        else:
            signal = norm_by_noisiest_section(signal,offset = offset)[0].astype(np.float32)
        if read_len <= overlay:
            current_chunks = [signal]
        else:
            current_chunks = [signal[i:i+chunk_len] for i in np.arange(0,read_len-overlay,chunk_len-overlay)]
        last_chunk = current_chunks[-1]
        last_chunk_len = len(last_chunk)
        current_chunks[-1]= np.pad(last_chunk,(0,chunk_len-last_chunk_len),'constant',constant_values = (0,0))
        chunks += current_chunks
        meta_info += [(fast5_f,read_id,chunk_len)]*(len(current_chunks)-1) 
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
    """
    Calculate the quality score for the consensus read.

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

def viterbi_decode(logits:torch.tensor):
    """
    Viterbi decdoing algorithm

    Parameters
    ----------
    logits : torch.tensor
        Shape L-N-C

    Returns
    -------
        sequence: A length N list contains final decoded sequence.
        moves: A length N list contains the moves array.

    """
    sequence = np.argmax(logits,axis = 2)
    sequence = sequence.T #L,N -> N,L
    sequence,moves = raw2seq(sequence)
    return sequence,list(moves.astype(int))

def consensus_call(seqs,assembly_func,metas,moves,qcs,strides,hiddens = None, logits = None,jump_ratio = None):
    if hiddens:
        curr = [x for x in zip(seqs,metas,qcs,moves,hiddens,logits) if x[1][1] == metas[0][1]]
        curr_seqs,curr_meta,qs_list,moves,hiddens,logits = zip(*curr)
        pad_length = len(moves[0]) - curr_meta[-1][2]//strides
        pad_length_move = len(moves[0]) - np.max([-1] + list(np.where(moves[-1])[0]))-1
        pad_length = min(pad_length,pad_length_move)
        # This pad length gurantee that sum(move) = len(seq), as sometime the base can be called in padding region.
        hiddens = np.asarray(hiddens)
        hiddens = hiddens.reshape((-1,hiddens.shape[-1]))[:-pad_length]
        logits = np.asarray(logits)
        logits = logits.reshape((-1,logits.shape[-1]))[:-pad_length]
    else:
        curr = [x for x in zip(seqs,metas,qcs,moves) if x[1][1] == metas[0][1]]
        curr_seqs,curr_meta,qs_list,moves = zip(*curr)
        pad_length = len(moves[0]) - curr_meta[-1][2]//strides
    css = assembly_func(curr_seqs,qs_list = qs_list,jump_step_ratio = jump_ratio)
    curr_move = np.asarray(moves).flatten()
    fast5f = set([x[0] for x in curr_meta])
    id = set([x[1] for x in curr_meta])
    assert len(fast5f)==1 and len(id)==1
    return css,id.pop(),fast5f.pop(),curr_move[:-pad_length],hiddens,logits

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
        self.hiddens = []
        self.logits = []
        self.count = 0
        self.config = config
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
        self.chunk_length = config.EVAL['chunk_len']
        
    def add(self,css,read_id,fast5f,move,hidden = None,logits = None):
        if(len(css)) == 2:
            self.css_seqs.append(list2string(np.argmax(css[0],axis=0),base_type = self.base_dict))
            self.qs.append(qs(css[0],css[1]))
        else:
            self.css_seqs.append(list2string(np.argmax(css,axis = 0),base_type = self.base_dict))
        self.moves.append(move)
        self.read_ids.append(read_id)
        self.fast5_fs.append(fast5f)
        if hidden is not None:
            self.hiddens.append(hidden)
        if logits is not None:
            self.logits.append(logits)
        if len(self.css_seqs) >= self.seq_batch:
            self.flush()
    
    def flush(self):
        if not len(self.css_seqs):
            #every thing has been flushed out.
            return 
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
        self.hiddens = []
        self.qs = []
        self.logits = []
        
    def write_fasta(self,output_f):
        with open(output_f,mode = 'w+') as f:
            for seq,name in zip(self.css_seqs,self.read_ids):
                f.write(">%s\n%s\n"%(name,seq))
    
    def write_fast5(self,output_f):
        last_dest = None
        root = None
        for seq,name,fast5,q,move,hidden,logits in zip(self.css_seqs,self.read_ids,self.fast5_fs,self.qs,self.moves,self.hiddens,self.logits):
            dest = os.path.join(output_f,os.path.basename(fast5))
            if not os.path.isfile(dest):
                shutil.copy(fast5,dest)
            if dest != last_dest:
                if root:
                    root.close()
                root = h5py.File(dest,mode='a')
            if 'read_%s'%(name) not in root:
                print("Warning read %s is not found in the fast5 %s"%(name,dest))
                print("An new entry will be created but original sequencing information will be lost.")
                root.create_group("read_%s"%(name))
            if 'read_%s/Analyses'%(name) in root:
                read_h = root['read_%s/Analyses'%(name)]
                existed_basecall = [x for x in list(read_h.keys()) if 'Basecall' in x]
            else:
                existed_basecall = []
                root['read_%s'%(name)].create_group('Analyses')
                read_h = root['read_%s/Analyses'%(name)]
            curr_group = read_h.create_group('Basecall_1D_%03d'%(len(existed_basecall)))
            curr_group.attrs.create('name','Xron')
            curr_group.attrs.create('time_stamp','%s'%(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))
            result_h = curr_group.create_group('BaseCalled_template')
            result_h.create_dataset("Fastq",data = "@%s\n%s\n+\n%s\n"%(name,seq,q))
            result_h.create_dataset("Move",data = move,dtype='i8')
            result_h.create_dataset("Hidden",data = hidden, dtype = "f")
            result_h.create_dataset("Logits",data = logits, dtype = "f")
            summary_h = curr_group.create_group("Summary")
            summary_h = summary_h.create_group("basecall_1d_template")
            summary_h.attrs.create('block_stride',self.stride)
            summary_h.attrs.create('chunk_length',self.chunk_length)
            last_dest = dest
        if root:
            root.close()
    
    def write_fastq(self,output_f):
        with open(output_f,mode = 'w+') as f:
            for seq,name,q in zip(self.css_seqs,self.read_ids,self.qs):
                f.write("@%s\n%s\n+\n%s\n"%(name,seq,q))
    
    def write_config(self,output_f):
        out_config = {"CTC":self.config.CTC,"EVAL":self.config.EVAL}
        config_f = os.path.join(output_f,'config.toml')
        with open(config_f,mode = 'w+') as f:
            toml.dump(out_config,f)

class Evaluator(object):
    def __init__(self,
                 net:CRNN,
                 config:CONFIG,
                 assembly_function:Callable,
                 writer:Writer):
        """

        Parameters
        ----------
        nets : CRNN
            A CRNN network instance.
        config: CONFIG
            A CONFIG class contains basecall configurations.
        assembly_function: Callable
            The assembly function used to assemble chunks into sequence
        writer: Writer:
            The writer function to write the output.
        Returns
        -------
        None.

        """
        self.net = net
        self.config = config
        self.result_collections = {'seqs':[],'metas':[],'moves':[],'qcs':[]}
        if config.EVAL['format'] == 'fast5':
            self.activation = {}
            def get_activation(name):
                def hook(model,input,output):
                    self.activation[name] = output.permute([1,0,2]).cpu().detach().numpy()
                return hook
            net.net[-4].register_forward_hook(get_activation('fnn1_out_linear'))
            self.result_collections['hiddens'] = []
            self.result_collections['logits'] = []
        self.elapse_time = {}
        self.assembly_time, self.nn_time, self.writing_time = 0.,0.,0.
        self.writer = writer
        self.assembly_func = assembly_function
    
    @property
    def seqs(self):
        return self.result_collections['seqs']
    
    @property
    def metas(self):
        return self.result_collections['metas']
    
    @property
    def moves(self):
        return self.result_collections['moves']
    
    @property
    #The quality scores: biggest logits
    def qcs(self):
        return self.result_collections['qcs']
    
    @property
    def logits(self):
        if 'logits' in self.result_collections.keys():
            return self.result_collections['logits']
        else:
            return None
    
    @property
    def hiddens(self):
        if 'hiddens' in self.result_collections.keys():
            return self.result_collections['hiddens']
        else:
            return None
        
    def _timeit(func):
        def wrap(self,*args,**kwargs):
            start = timer()
            func(self,*args,**kwargs)
            if func not in self.elapse_time.keys():
                self.elapse_time[func] = timer() - start
            else:
                self.elapse_time[func] += timer() - start
        return wrap
        
    @_timeit
    def run_nn(self,batch):
        start = timer()
        self.curr_logits = self.net(batch).cpu().numpy() #TODO send the logits into a queue to enable multi-threading
        self.nn_time += timer() - start
    
    def beam_decode(self,posteriors):
        L,N,C = posteriors.shape
        move = [np.zeros(L,dtype = int) for _ in np.arange(N)]
        seqs = []
        for i in np.arange(N):
            p = np.exp(posteriors[:,i,:])
            seq,path = beam_search(p,
                                   "N"+self.config.CTC["alphabeta"],
                                   beam_size = self.config.CTC["beam"],
                                   beam_cut_threshold = self.config.CTC["beam_cut_threshold"])
            move[i][path] = 1
            seqs.append([self.config.CTC["alphabeta"].index(x) for x in seq])
        return seqs,move
    
    def run_once(self,batch,meta):
        self.run_nn(batch)
        start = timer()
        if self.config.CTC["beam"] <= 1:
            sequence,move = viterbi_decode(self.curr_logits)
        else:
            sequence,move = self.beam_decode(self.curr_logits)
        self.result_collections['qcs'] += list(np.max(self.curr_logits,axis = 2).T)
        self.result_collections['seqs'] += sequence
        self.result_collections['metas'] += meta
        self.result_collections['moves'] += move
        if self.config.EVAL['format'] == 'fast5':
            self.result_collections['hiddens'] += list(self.activation['fnn1_out_linear'])
            self.result_collections['logits'] += list(np.transpose(self.curr_logits,(1,0,2)))
        self.assembly_time += timer() - start
        self._call()
    
    def _call(self):
        if (None,None,None) in self.metas:
            print("Remove the padding chunks.")
            self._update_remain()
        while not all_equal([x[1] for x in self.metas]):
            self.call_once()
    @_timeit
    def call_once(self):
        start = timer()
        if len(self.seqs) == 0:
            return
        r = consensus_call(self.seqs,
                           self.assembly_func,
                           self.metas,
                           self.moves,
                           qcs = self.qcs,
                           strides = self.config.CNN['Layers'][-1]['stride'], 
                           hiddens = self.hiddens, 
                           logits = self.logits,
                           jump_ratio= 1-float(self.config.EVAL['overlay'])/self.config.EVAL['chunk_len'])
        self.assembly_time += timer() - start
        start = timer()
        self.writer.add(*r)
        writing_duration = timer()-start
        self.writing_time += writing_duration
        self._update_remain(key = self.metas[0][1])
    
    @_timeit
    def _update_remain(self,key = None):
        start = timer()
        remain = [x for x in zip(*list(self.result_collections.values())) if x[1][1] != key] #Update based on metas name (x[1] = metas)
        remain = list(zip(*remain))
        if len(remain) == 0:
            return
        for i,k in enumerate(list(self.result_collections.keys())):
            self.result_collections[k] = list(remain[i])
        self.assembly_time += timer() - start
            
def main(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    class CTC_CONFIG(CONFIG):
        CTC = {"beam_cut_threshold":0.05,
               "alphabeta": "ACGTM",
               "mode":"rna",
               "beam":args.beam}
    class CALL_CONFIG(CTC_CONFIG):
        EVAL = {"batch_size":args.batch_size,
                "chunk_len":args.chunk_len,
                'device':args.device,
                'assembly_method':args.assembly_method,
                'seq_batch':4000,
                'mode':'rna',
                'format':'fast5' if args.fast5 else 'fastq',
                'offset':args.offset,
                'diff_norm':args.diff_norm}        #Diffnorm is deprecated setting this to True will have no effect
    config = CALL_CONFIG()
    print("Construct and load the model.")
    model_f = args.model_folder
    config_old = load_config(os.path.join(model_f,"config.toml"))
    config_old.EVAL = config.EVAL #Overwrite training config.
    config_old.CTC = config.CTC
    config = config_old
    if args.config:
        config = load_config(args.config)
    net = CRNN(config)
    load(args.model_folder,net,device = args.device)
    print("Begin basecall.")
    net.eval()
    boostnano_evaluator = None
    if config.EVAL['mode'] == 'rna' or config.EVAL['mode'] == 'rna-meth':
        if args.boostnano:
            print("Loading BoostNano model.")
            project_f = os.path.dirname(os.path.dirname(inspect.getfile(CSM)))
            model_f = os.path.join(project_f,'BoostNano','model')
            boostnano_net = CSM()
            boostnano_evaluator = evaluator(boostnano_net,model_f)
    config.EVAL['overlay'] = int(args.overlay_ratio * args.chunk_len)
    df = chunk_feeder(args.input, config, boostnano_evaluator)
    writer = Writer(args.output,config)
    writer.write_config(args.output)
    assembly_func = partial(simple_assembly_qs,
                            kernal = args.assembly_method,
                            alphabeta = config.CTC["alphabeta"])
    caller = Evaluator(net,config,assembly_function = assembly_func,writer = writer)
    with torch.no_grad():
        for chunk,meta in df:
            caller.run_once(chunk,meta)
    start_w = timer()
    caller.call_once()
    writer.flush()
    caller.writing_time += timer()-start_w
    print("NN_time:%f,assembly_time:%f,writing_time:%f"%(caller.nn_time,caller.assembly_time,caller.writing_time))

def add_arguments(parser):
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
                        help="Evaluation batch size, default use the maximum size of the GPU memory, batch size and memory consume relationship: 200:6.4GB, 400:11GB, 800:20GB, 1200:30GB.")
    parser.add_argument('--chunk_len', default = 2000, type = int,
                        help="The length of each chunk signal.")
    parser.add_argument('--overlay_ratio', default = 0.0, type = float,
                        help="The ratio of the overlay between two chunks.")
    parser.add_argument('--config', default = None,
                        help = "Training configuration.")
    parser.add_argument('--threads', type = int, default = None,
                        help = "Number of threads used by Pytorch")
    parser.add_argument('--assembly_method',type = str, default = "stick",
                        help = "Assembly method used, can be glue, global, simple and stick.")
    parser.add_argument('--offset',type = float,default = 0.0,
                        help = "Manual set a offset to the normalized signal.")
    parser.add_argument('--diff_norm', action="store_true", dest="diff_norm",
                        help = "Turn on the differential normalization.")
    parser.add_argument('--beam', type = int, default = 1,
                        help = "Beam size of the beam search decoder, default\
                        is 1 where Viterbi decoder is used.")
    parser.add_argument('--boostnano', action="store_true", dest="boostnano",
                        help = "Enable boostnano preprocessing.")

def post_args(args):
    MEMORY_PER_BATCH_PER_SIGNAL=15000. #B
    if args.batch_size is None:
        if torch.cuda.is_available():
            t = torch.cuda.get_device_properties(0).total_memory
            args.batch_size = int(t/MEMORY_PER_BATCH_PER_SIGNAL/args.chunk_len//100*100)
            print("Auto configure to use %d batch_size for a total of %.1f GB memory."%(args.batch_size,t/1024**3))
        else:
            args.batch_size = 1200
            print("No GPU is detected, the batch_size is setting to default %d"%(args.batch_size))
    os.makedirs(args.output,exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Call a xron model on dataset.')
    add_arguments(parser)
    args = parser.parse_args(sys.argv[1:])
    post_args(args)
    main(args)
