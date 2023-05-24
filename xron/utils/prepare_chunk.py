"""
Created on Wed Apr 28 23:03:36 2021

@author: Haotian Teng
"""

import os
import sys
import h5py
import toml
import numpy as np
import argparse
from matplotlib import pyplot as plt
from itertools import compress
import seaborn as sns
from tqdm import tqdm
from xron.utils.seq_op import fast5_iter,norm_by_noisiest_section,diff_norm_by_noisiest_section,diff_norm_fixing_deviation
from xron.utils.align import MetricAligner
from Bio.Seq import Seq
from functools import partial
from boostnano.boostnano_model import CSM
from boostnano.boostnano_eval import evaluator
import inspect
import os
from pathlib import Path
from xron.nrhmm.prepare_data import Extractor

alt_map = {'ins':'0','M':'A','U':'T'}
complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} 
MIN_READ_SEQ_LEN = 100 #The filter of minimum sequence length.
RNA_FILTER_CONFIG = {"min_rate":25,
                     "min_seq_len":3, #this is in term of each chunk
                     "max_gap_allow":2000,
                     "min_quality_score":0.85, #the minimum quality score for a chunk to be included into traning set.
                     "max_mono_prop":0.6}

DNA_FILTER_CONFIG = {"min_rate":2,
                     "min_seq_len":7,
                     "max_gap_allow":400,
                     "min_quality_score":0.8, #the minimum quality score for a chunk to be included into traning set.
                     "max_mono_prop":0.8}
def reverse_complement(seq):    
    return str(Seq(seq).reverse_complement())

def chop(arr,chunk_length,padding = True,pad_values = 0):
    read_len = len(arr)
    chunks = np.split(arr,np.arange(0,read_len,chunk_length))[1:]
    length = [len(x) for x in chunks]
    if padding:
        last_chunk = chunks[-1]
        chunks[-1]= np.pad(last_chunk,(0,chunk_length-len(last_chunk)),'constant',constant_values = (pad_values,pad_values))
    return chunks,length

def clean_repr(seq):
    for k,v in alt_map.items():
        seq = seq.replace(k,v)
    return seq

def retrive_seq(seq_h,event_stride):
    moves = np.asarray(seq_h['BaseCalled_template']['Move'])
    try:
        seq = np.asarray(seq_h['BaseCalled_template']['Fastq']).tobytes().decode('utf-8').split('\n')[1]
    except:
        seq = str(np.asarray(seq_h['BaseCalled_template']['Fastq']).astype(str)).split('\n')[1]
    pos = np.repeat(np.cumsum(moves)-1,repeats = event_stride).astype(np.int32)
    return seq,pos
    
def filt(filt_config,chunks,seq,seq_len,durations,meta,*args):
    n = chunks.shape[0]
    segment_len = chunks.shape[1]
    print("Origin %d chunks in total."%(chunks.shape[0]))
    
    ### Filter out by length
    max_seq_len = np.int(segment_len/filt_config['min_rate'])
    mask_min = seq_len>filt_config["min_seq_len"]
    mask_max = seq_len<max_seq_len
    ### Filter by quality score
    qs_mask = np.asarray([x!='X' for x in seq]) # 'X' is the mark of low quality chunks.
    qs_penc = 100*sum(np.logical_not(qs_mask))/n
    print("%.2f"%(qs_penc),"% chunks has been filtered out because of quality score.")

    ### Filter by boostnano
    bn_mask = np.asarray([x!='Y' for x in seq]) # 'Y' means current chunks is from a polyA tail or adapter sequence.
    bn_penc = 100*sum(np.logical_not(bn_mask))/n
    print("%.2f"%(bn_penc),"% chunks has been filtered out because they are polyA tail or adapter.")    
    
    ### Filter by gap
    gap_mask = np.asarray([x!='P' for x in seq]) # 'P' is the mark of chunks with large gap.
    gap_perc = 100*sum(np.logical_not(gap_mask))/n
    print("%.2f"%(gap_perc),"% chunks has been filtered out because of large gap.")

    print("%.2f"%(100*sum(np.logical_not(mask_min))/n - gap_perc - qs_penc - bn_penc),"% chunks are filted out by minimum sequence length filter.")
    print("%.2f"%(100*sum(np.logical_not(mask_max))/n),"% chunks are filted out by maximum sequence length filter.")
    mask = (mask_min*mask_max*qs_mask*gap_mask*bn_mask).astype(bool)
    
    
    ### Filter out by monopolization
    mono_mask = np.asarray([max(np.unique(list(x),return_counts = True)[1])/y<filt_config['max_mono_prop'] for x,y in zip(seq[mask],seq_len[mask])])
    print("%.2f"%(100*sum(np.logical_not(mono_mask))/sum(mask)),"% chunks are filted out by mono filter.")
    ### Filter by unrecognized nucleotide type
    type_mask = np.asarray([len(x.replace('A','').replace('G','').replace('C','').replace('T',''))==0 for x in seq[mask]])
    print("%.2f"%(100*sum(np.logical_not(type_mask))/sum(mask)),"% chunks are filted out by nucleotide filter.")
    
    mono_mask = (mono_mask*type_mask).astype(bool) 
    
    pass_chunks = chunks[mask][mono_mask]
    print("In total %.2f"%(100*len(pass_chunks)/len(chunks)),"% of reads pass the filters.")
    filtered_args = [arg[mask][mono_mask] for arg in args]
    if filtered_args:
        return pass_chunks,seq[mask][mono_mask],seq_len[mask][mono_mask],durations[mask][mono_mask],list(compress(compress(meta,mask),mono_mask)),filtered_args
    else:
        return pass_chunks,seq[mask][mono_mask],seq_len[mask][mono_mask],durations[mask][mono_mask],list(compress(compress(meta,mask),mono_mask))
    
def rna_filt(chunks,seq,seq_len,durations,meta,*args):
    return partial(filt,RNA_FILTER_CONFIG)(chunks,seq,seq_len,durations,meta,*args)

def dna_filt(chunks,seq,seq_len,durations,meta,*args):
    return partial(filt,DNA_FILTER_CONFIG)(chunks,seq,seq_len,durations,meta,*args)

def extract(args):
    if args.mode == 'dna':
        FILTER_CONFIG = DNA_FILTER_CONFIG
    else:
        FILTER_CONFIG = RNA_FILTER_CONFIG
    if args.write_correction:
        m = 'r+'
    else:
        m = 'r'
    if args.extract_kmer:
        extractor = Extractor(k=args.kmer_size,alphabeta = args.alphabeta)
    iterator = fast5_iter(args.input_fast5,mode = m)
    if args.diff_sig:
        if args.config['fixed_deviation']:
            norm_func = diff_norm_fixing_deviation
        else:
            norm_func = diff_norm_by_noisiest_section
    else:
        norm_func = norm_by_noisiest_section
    if args.extract_seq:
        print("Read reference genome.")
        aligner = MetricAligner(args.reference,options = '-x ont2d')
    print("Loading BoostNano model.")
    project_f = os.path.dirname(os.path.dirname(inspect.getfile(CSM)))
    model_f = os.path.join(project_f,'BoostNano','model')
    net = CSM()
    boostnano_evaluator = evaluator(net,model_f)
    print("Begin processing the reads.")
    meta_info,chunks,seqs,meds,mads,meths,offsets,scales,read_ids,kmers,durations = [],[],[],[],[],[],[],[],[],[],[]
    if args.mode == "rna" or args.mode == "rna_meth":
        reverse_sig = True
    fail_read_count = {"No basecall":0,
                       "Alignment failed":0,
                       "Read too short":0,
                       "Sequence length is inconsistent with signal length":0,
                       "Kmer extraction failed":0,
                       "Processed":0,}
    
    ### Debug code
    qss = []
    ###
    loop_obj = tqdm(iterator)
    for read_h,signal,fast5_f,read_id in loop_obj:
        postfix_str = "no entry: %d, alignment failed: %d, read too short: %d, inconsistent length: %d, kmer fail:%d, processed:%d"%(fail_read_count['No basecall'],
                                                                                                                                  fail_read_count['Alignment failed'],
                                                                                                                                  fail_read_count['Read too short'],
                                                                                                                                  fail_read_count['Sequence length is inconsistent with signal length'],
                                                                                                                                  fail_read_count['Kmer extraction failed'],
                                                                                                                                  fail_read_count['Processed'])
        loop_obj.set_postfix_str(postfix_str)
        read_len = len(signal)
        original_signal = signal.astype(np.float32)
        signal,med,mad = norm_func(signal)
        signal = signal.astype(np.float16)
        if reverse_sig:
            signal = signal[::-1]
        if args.extract_seq: 
            try:
                seq, pos = retrive_seq(read_h['Analyses/Basecall_1D_%s'%(args.basecall_entry)],
                                   args.stride)
                basecall_entry = args.basecall_entry
            except KeyError:
                if args.alternative_entry is not None:
                    try:
                        seq, pos = retrive_seq(read_h['Analyses/Basecall_1D_%s'%(args.alternative_entry)],
                                           args.stride)
                        basecall_entry = args.alternative_entry
                    except KeyError:
                        print("No basecall information was found in entry Basecall_1D_%s of read %s of %s, skip the read."%(args.basecall_entry,read_id,fast5_f))
                        fail_read_count["No basecall"]+=1
                        continue
                else:
                    print("No basecall information was found in entry Basecall_1D_%s of read %s of %s and no alternative entry is set, skip the read."%(args.basecall_entry,read_id,fast5_f))
                    fail_read_count["No basecall"]+=1
                    continue
            assert read_h['Analyses/Basecall_1D_%s/Summary/basecall_1d_template'%(basecall_entry)].attrs['block_stride'] == args.stride
            if args.write_correction:
                if not "Segmentation_%s"%(basecall_entry) in read_h['Analyses']:
                    read_h['Analyses/'].create_group("Segmentation_%s"%(basecall_entry))
                if not "Reference_corrected" in read_h['Analyses/Segmentation_%s/'%(basecall_entry)]:
                    read_h['Analyses/Segmentation_%s/'%(basecall_entry)].create_group("Reference_corrected")
                else:
                    del read_h['Analyses/Segmentation_%s/Reference_corrected'%(basecall_entry)]
                    read_h['Analyses/Segmentation_%s/'%(basecall_entry)].create_group("Reference_corrected")
            if len(seq) < MIN_READ_SEQ_LEN:
                fail_read_count["Read too short"]+=1
                continue
            if seq.count('A')+seq.count('M') == 0:
                meths.append(np.nan)
            else:
                meths.append(seq.count('M')/(seq.count('A')+seq.count('M')+1e-8))
            seq = clean_repr(seq) #M->A, U->T
            hits,ref_seq,ref_idx,qs = aligner.ref_seq(seq,exclude_negative=True) #ref_idx: the ith ref_idx R_i means the ith base in reference sequence is the R_i base in basecalled sequence
            if not hits:
                fail_read_count['Alignment failed'] += 1
                continue
            if args.mode == 'rna' or args.mode == 'rna_meth':
                (decoded,path,locs) = boostnano_evaluator.eval_sig(original_signal,1000)
                locs = read_len - locs
            assert np.all(np.diff(ref_idx)>=0)
            try:
                start = int(read_h['Analyses/Segmentation_%s/Summary/segmentation'%(basecall_entry)].attrs['first_sample_template'])
            except:
                start = 0
            if reverse_sig and args.rev_move and start>0:
                signal = signal[:-start]
            if args.rev_move and reverse_sig:
                pos = pos[::-1]
                pos = pos[0] - pos 
            else:
                signal = signal[start:]
            if abs(len(signal)-len(pos)) > (min(len(signal),len(pos))/args.stride):
                print(fast5_f,read_id,len(pos),len(signal))
                print("The signal length is %d and position length is (%d) for read %s of %s, check if the stride is correct."%(len(signal),len(pos),read_id,fast5_f))
                fail_read_count["Sequence length is inconsistent with signal length"] +=1
                continue
            if len(signal)>len(pos):
                if args.padding:
                    #Insert values on random selected indexs.
                    padding = len(signal) - len(pos)
                    idxs = np.random.choice(len(pos),padding,replace = False)
                    pos = np.insert(pos,idxs,pos[idxs])
                else:
                    signal = signal[:len(pos)]
            else:
                pos = pos[:len(signal)]
            if len(signal) == 0:
                continue
            read_len = len(pos)
            if args.write_correction:
                ref_idx_aligned = ref_idx[ref_idx<=pos[-1]]
                ref_seq_aligned = ref_seq[:len(ref_idx_aligned)]
                if 'N' in ref_seq_aligned:
                    fail_read_count["Kmer extraction failed"] += 1
                    continue
                qs_aligned = qs[:len(ref_idx_aligned)]
                ref_sig_idx = [np.where(pos == x)[0][0] for x in ref_idx_aligned] #The ith ref_sig_idx REF_i means the ith base in reference sequence is the REF_i signal point at the reversed signal.
                read_h['Analyses/Segmentation_%s/Reference_corrected'%(basecall_entry)].create_dataset("ref_sig_idx",data = ref_sig_idx)
                read_h['Analyses/Segmentation_%s/Reference_corrected'%(basecall_entry)].create_dataset("ref_seq",data = ref_seq_aligned)
                read_h['Analyses/Segmentation_%s/Reference_corrected'%(basecall_entry)].create_dataset("map_score",data = qs_aligned)
                if args.extract_kmer:
                    try:
                        _,kmer_seqs = extractor.kmer_decode(ref_seq_aligned, signal, np.asarray(ref_sig_idx), padded = True)
                    except ValueError:
                        fail_read_count["Kmer extraction failed"]+=1
                        continue
            fail_read_count["Processed"] += 1
            for x in np.arange(0,read_len,args.chunk_len):
                if args.mode == "rna" or args.mode == "rna_meth":
                    if x+args.chunk_len > locs[0]:
                        seqs.append('Y')
                        continue
                s,e = pos[x:x+args.chunk_len][[0,-1]]
                mask = (ref_idx>=s)&(ref_idx<=e)
                qs_mask = qs[mask]
                if sum(mask) > 0:
                    qss.append(sum(qs_mask)/len(qs_mask))
                    if sum(qs_mask)/len(qs_mask) <= FILTER_CONFIG["min_quality_score"]:
                        seqs.append('X')
                        continue
                    if np.max(np.unique(pos[x:x+args.chunk_len],return_counts = True)[1])>FILTER_CONFIG["max_gap_allow"]:
                        seqs.append('P')
                        continue
                    r_s,r_e = np.where(mask)[0][[0,-1]]
                    # print("Basecall sequence:%s"%(basecall_seq[s:e+1]))
                    seq = ref_seq[r_s:r_e+1]
                    # print("Aligned seqeuence:%s"%(seq))
                    if args.mode == 'rna-meth':
                        seq.replace('A','M')
                    elif args.mode == 'rna':
                        seq.replace('M','A')
                    seqs.append(seq)
                else:
                    seqs.append('')
        meds.append(med)
        mads.append(mad)
        if 'channel_id' in read_h:
            offsets.append(read_h['channel_id'].attrs['offset'])
            scales.append(read_h['channel_id'].attrs['range'])
        else:
            offsets.append(read_h['UniqueGlobalKey/channel_id'].attrs['offset'])
            scales.append(read_h['UniqueGlobalKey/channel_id'].attrs['range'])
        read_ids.append(read_id)
        curr_chunks,curr_duration = chop(signal,args.chunk_len,pad_values = 0)
        if args.extract_kmer:
            curr_kmers,_ = chop(kmer_seqs,args.chunk_len,pad_values = -1)
            kmers += curr_kmers
        chunks += curr_chunks
        durations += curr_duration
        meta_info += [(fast5_f,read_id,str(args.chunk_len),str(args.stride))]*len(curr_chunks)
        if args.max_n and (args.max_n > 0) and (len(chunks)>args.max_n):
            chunks = chunks[:args.max_n]
            seqs = seqs[:args.max_n]
            meta_info = meta_info[:args.max_n]
            kmers = kmers[:args.max_n]
            durations = durations[:args.max_n]
            break
    for key,val in fail_read_count.items():
        print(key,':',val)
    if len(chunks) == 0:
        raise ValueError("No chunk is added to dataset, check the setting.")
    chunks = np.stack(chunks,axis = 0)
    durations = np.asarray(durations,dtype = np.int)
    if args.extract_kmer:
        kmers = np.stack(kmers,axis = 0)
    print("Average median value %f"%(np.mean(meds)))
    print("Average median absolute deviation %f"%(np.mean(mads)))
    print("Average methylation proportion %f"%(np.nanmean(meths)))
    if args.extract_seq:
        seq_lens = [len(i) for i in seqs]
        seqs = np.array(seqs)
        seq_lens = np.array(seq_lens)
        filt_func = dna_filt if args.mode == "dna" else rna_filt
        if args.extract_kmer:
            chunks,seqs,seq_lens,durations,meta_info,more = filt_func(chunks,seqs,seq_lens,durations,meta_info,kmers)
            kmers = more[0]
        else:
            chunks,seqs,seq_lens,durations,meta_info = filt_func(chunks,seqs,seq_lens,durations,meta_info)
        np.save(os.path.join(args.output,'seqs.npy'),seqs)
        np.save(os.path.join(args.output,'seq_lens.npy'),seq_lens)
    if args.extract_kmer:
        np.save(os.path.join(args.output,'kmers.npy'),kmers)
    np.save(os.path.join(args.output,'mm.npy'),(mads,meds,meths,offsets,scales,read_ids))
    np.savetxt(os.path.join(args.output,'meta.csv'),meta_info,fmt="%s")
    np.save(os.path.join(args.output,'chunks.npy'),chunks)
    np.save(os.path.join(args.output,'durations.npy'),durations)
    config_file = os.path.join(args.output,'config.toml')
    config_modules = [x for x in args.__dir__() if not x .startswith('_')][::-1]
    config_dict = {x:getattr(args,x) for x in config_modules}
    config_dict['FILTER_CONFIG'] = FILTER_CONFIG
    config_dict['minimum_sequence_length']:MIN_READ_SEQ_LEN
    config_dict["chunk_len"] = args.chunk_len
    if args.extract_kmer:
        config_dict["k"] = args.kmer_size
        config_dict["alphabeta"] = args.alphabeta
        config_dict['kmer2idx_dict'] = extractor.kmer2idx_dict
        config_dict['idx2kmer'] = extractor.idx2kmer
    plt.figure()
    sns.distplot(qss)
    plt.xlabel("Quality score")
    plt.ylabel("Distribution")
    plt.savefig(os.path.join(args.output,"qs_dist.png"))
    with open(config_file,'w+') as f:
        toml.dump(config_dict,f)

def add_arguments(parser):
    parser.add_argument('-i', 
                        '--input_fast5', 
                        required = True,
                        help="File path or Folder path to the fast5 file.")
    parser.add_argument('-o',
                        '--output',
                        required = True,
                        help="Output folder.")
    parser.add_argument('--chunk_len',
                        default = 4000,
                        type=int,
                        help="The lenght of the segment in chunk.")
    parser.add_argument('--max_n',
                        default = None,
                        type=int,
                        help="The maximum number of the segments to be extracted")
    parser.add_argument("--extract_seq",
                        action = "store_true",  
                        help = "If the sequence information is going to be\
                            extracted.")
    parser.add_argument("--extract_kmer",
                        action = "store_true",
                        help = "If the kmer information is going to be extracted.")
    parser.add_argument("--kmer_size","-k",
                        default = None,
                        type = int,
                        help = "The length of the kmer to be extracted.")
    parser.add_argument("--write_correction",
                        action = "store_true",
                        dest = "write_correction",
                        help = "If write correcting alignment to the original fast5 file.")
    parser.add_argument('--basecall_entry',
                        default = "000",
                        help="The entry number in /Analysis/ to look into, for\
                            example 000 means looking for Basecall_1D_000.")
    parser.add_argument('--alternative_entry',
                        default = None,
                        type = str,
                        help="If basecall information is not found in the basecall entry, look into this alternative entry.")
    parser.add_argument('--stride',
                        default = 12,
                        type = int,
                        help = "The length of stride used in basecall model,\
                        for guppy RNA fast, this number is 12, for guppy RNA\
                        hac model, this number is 10, for xron this number is\
                        5 or 11."
                        )
    parser.add_argument('--padding',
                        type = bool,
                        default = True,
                        help = "Padding the position to make the position has same length, otherwise cut the signal.")
    parser.add_argument('--move_direction',
                        default = 'forward',
                        help = "The direction of the output Move matrix, for Guppy, it's forward, for Xron it's backward.")
    parser.add_argument('--basecaller',
                        default = None,
                        help = "The basecaller setting, default is None, can be xron, guppy and guppy_fast, set this argument will override args.stride and args.move_direction.")
    parser.add_argument('--reference',
                        default = None,
                        help = "The reference genome, it's required when\
                        extract_seq is set to True")
    parser.add_argument('--mode',
                        default = "rna",
                        help = "Can be one of this mode: rna, dna, rna-meth")
    parser.add_argument('--fix_d',action="store_true",
                        dest = "fix_d",
                        help = "Use a fix deviation to normalize the signal.")

def post_args(FLAGS):
    XRON_CONFIG = {"stride":11,
                   "padding":True, #If we padding the position to the same length as signal.
                   "differential_signal":False,
                   "forward_move_matrix":False,#If the move matrix is count on backward signal or forward signal, True means the move matrix is along the forward signal, so the move matrix should be reveresed first as the signal is reveresed before processing.
                   "fixed_deviation":FLAGS.fix_d} 
    GUPPY_CONFIG = {"stride":10,
                    "chunk_len":2000, #The length of the chunk when basecall.
                    "padding":False,
                    "forward_move_matrix":True,
                    "differential_signal":False}
    GUPPY_FAST_CONFIG = {"stride":12,
                         "padding":False,
                         "chunk_len":2000, #The length of the chunk when basecall.
                         "forward_move_matrix":True,
                         "differential_signal":False}
    config_dict = {"xron":XRON_CONFIG,
                   "guppy":GUPPY_CONFIG,
                   "guppy_fast":GUPPY_FAST_CONFIG}
    if FLAGS.basecaller:
        config = config_dict[FLAGS.basecaller]
        FLAGS.stride = config["stride"]
        FLAGS.rev_move = config["forward_move_matrix"]
        FLAGS.diff_sig = config["differential_signal"]
        FLAGS.padding = config['padding']
    else:
        print("Basecaller is not defined, initialize the configuration with guppy fast model.")
        config = config_dict["guppy_fast"]
        FLAGS.rev_move = FLAGS.move_direction
    FLAGS.config = config
    FLAGS.alphabeta = "ACGTM" if FLAGS.mode == "rna_meth" else "ACGT"
    assert FLAGS.mode in ["rna","dna","rna_meth"], "Mode can only be one of rna, dna, rna_meth"
    if FLAGS.extract_kmer:
        assert FLAGS.kmer_size is not None, "Please specify the kmer length when extract kmer is enable."
        assert FLAGS.write_correction, "Please enable write_correction when extract kmer is enable."
    if FLAGS.write_correction:
        assert FLAGS.reference is not None, "Please specify the reference genome when write_correction is enable."
        assert FLAGS.extract_seq, "Please enable extract_seq when write_correction is enable."
    if FLAGS.extract_seq:
        if not FLAGS.reference:
            raise ValueError("Reference genome is required when extract the \
                             sequence.")
    os.makedirs(FLAGS.output,exist_ok = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='xron',   
                                     description='A Unsupervised Nanopore basecaller.')
    add_arguments(parser)
    FLAGS = parser.parse_args(sys.argv[1:])
    post_args(FLAGS)
    extract(FLAGS)


