# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Tue May  2 15:39:29 2017

from __future__ import absolute_import
from __future__ import print_function
import difflib
import math
import operator
import time
from xron.utils.seq_op import list2string,string2list
from collections import Counter
from itertools import groupby
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest
import numpy as np
import six
from six.moves import range
from Bio import pairwise2

def mapping(full_path, blank_pos=4):
    """Perform a many to one mapping in the CTC paper, merge the repeat and remove the blank
    Input:
        full_path:a vector of path, e.g. [1,0,3,2,2,3]
        blank_pos:The number regarded as blank"""
    full_path = np.asarray(full_path)
    merge_repeated = np.asarray([k for k, g in groupby(full_path)])
    blank_index = np.argwhere(merge_repeated == blank_pos)
    return np.delete(merge_repeated, blank_index)

def group_consecutives(vector, step=1):
    group = list()
    group_list = list()
    expect = None
    for x in vector:
        if (x != expect) and (expect is not None):
            group_list.append(group)
            group = []
        group.append(x)
        expect = x + step
    group_list.append(group)
    return group_list


###########################Section decoding method#############################
def section_decoding(logits, blank_thres=0.6, base_type=0):
    """Implemented the decoding method described in ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
    Find the best path between the section that divided by blank logits < 0.9
    
    logits: [batch_size,seg_length,neucloe_type+1]
    base_type: 0:dna 1:methylation 2:rna
    """
    prob = np.exp(logits) / (np.sum(np.exp(logits), axis=2)[:, :, None])
    if base_type == 0:
        blank_pos = 4
    mask = prob[:, :, blank_pos] < blank_thres
    batch_size, seg_len, nc_type = prob.shape
    bpreads = list()
    bpread = list()
    for batch_i in range(batch_size):

        group_list = group_consecutives(np.where(mask[batch_i, :])[0])

        bpread = []
        for group in group_list:
            if len(group) == 0:
                continue
            bpread.append(4)
            #            most_prob_path = best_path(prob[batch_i,group,:],base_type = base_type)
            most_mc_path = mc_path(logits[batch_i, group, :],
                                   base_type=base_type)
            most_prob_path = string2list(most_mc_path[0])
            bpread += most_prob_path
        bpreads.append(list2string(mapping(bpread), base_type=base_type))
    return (bpreads)


def best_path(logits, base_type):
    """Enumerate decoder,*slow*"""
    T, base_num = logits.shape
    accum_prob = {}
    for i in range(base_num ** T):
        prob = 1
        index_list = []
        for j in range(T):
            index = i / base_num ** j % base_num
            prob *= logits[j, index]
            index_list.append(index)
        index_list = mapping(index_list)
        if len(index_list) > 0:
            key = list2string(index_list, base_type=base_type)
            accum_prob.setdefault(key, 0)
            accum_prob[key] += prob
    most_prob_path = max(six.iteritems(accum_prob), key=operator.itemgetter(1))[
        0]
    return string2list(most_prob_path, base_type=base_type)


def mc_path(logits, base_type, sample_n=300):
    """Manto Carlo decoder
    Input Args:
        logits:[T,base_num]
        base_tyep: 0:normal dna+blank
        sample_n: Times of sample used in the Manto Carlo simulation.
    """
    logits_shape = logits.shape
    bpreads = list()
    qc_score = list()
    prob = np.exp(logits) / (np.sum(np.exp(logits), axis=2)[:, :, None])
    base_num = logits_shape[-1]
    T = logits_shape[-2]
    interval = np.zeros((T, base_num))
    interval[:, 0] = prob[:, 0]
    for i in range(1, base_num - 1):
        interval[:, i] = interval[:, i - 1] + prob[:, i]
    interval[:, 4] = 1

    sample_index = np.zeros((sample_n, T))
    sample = np.random.random((sample_n, T))
    for j in range(T):
        sample_index[:, j] = np.searchsorted(interval[i, j, :], sample[:, j],
                                             side='left')
    merge_path = list()
    for repeat_i in range(sample_n):
        ###This step may be slow, considering implemented in C
        temp_path = mapping(sample_index[repeat_i, :])
        ###
        merge_path.append(list2string(temp_path, base_type=base_type))
    path_count = Counter(merge_path)
    print(path_count)
    max2path = path_count.most_common(2)
    p1 = max2path[0][1] / float(sample_n)
    p2 = max2path[1][1] / float(sample_n)
    qc_score.append(10 * math.log10(p1 / p2))
    bpreads.append(max2path[0][0])
    return bpreads


def mc_decoding(logits, base_type, sample_n=300):
    """Manto Carlo decoder
    Input Args:
        logits:[batch_size,T,base_num] or [T,base_num]
        base_tyep: 0:normal dna+blank
        sample_n: Times of sample used in the Manto Carlo simulation.
    """
    logits_shape = logits.shape
    if len(logits_shape) == 2:
        logits = [logits]
        batch_size = 1
    else:
        batch_size = logits_shape[0]
    bpreads = list()
    qc_score = list()
    prob = np.exp(logits) / (np.sum(np.exp(logits), axis=2)[:, :, None])
    base_num = logits_shape[-1]
    T = logits_shape[-2]
    interval = np.zeros((batch_size, T, base_num))
    interval[:, :, 0] = prob[:, :, 0]
    for i in range(1, base_num - 1):
        interval[:, :, i] = interval[:, :, i - 1] + prob[:, :, i]
    interval[:, :, 4] = 1

    sample_index = np.zeros((sample_n, T))
    for i in range(batch_size):
        print(i)
        sample = np.random.random((sample_n, T))
        for j in range(T):
            sample_index[:, j] = np.searchsorted(interval[i, j, :],
                                                 sample[:, j], side='left')
        merge_path = list()
        for repeat_i in range(sample_n):
            ###This step may be slow, considering implemented in C
            temp_path = mapping(sample_index[repeat_i, :])
            ###
            merge_path.append(list2string(temp_path, base_type=base_type))
        path_count = Counter(merge_path)
        print(path_count)
        max2path = path_count.most_common(2)
        p1 = max2path[0][1] / float(sample_n)
        p2 = max2path[1][1] / float(sample_n)
        qc_score.append(10 * math.log10(p1 / p2))
        bpreads.append(max2path[0][0])
    return bpreads


###############################################################################

#########################Simple assembly method################################
def simple_assembly_kernal(bpread, prev_bpread,error_rate, jump_step_ratio):
    """
    Kernal function of the assembly method.
    log_P ~ x*log((N*n1/L)) - log(x!) + Ns * log(P1/0.25) + Nd * log(P2/0.25)
    bpread: current read.
    prev_bpread: previous read.
    error_rate: Average basecalling error rate.
    jump_step_ratio: Jump step/Segment len
    """
    back_ratio = 6.5 * 10e-4
    p_same = 1 - 2*error_rate + 26/25*(error_rate**2)
    p_diff = 1 - p_same
    ns = dict() # number of same base
    nd = dict()
    log_px = dict()
    N = len(bpread)
    match_blocks = difflib.SequenceMatcher(a=bpread,b=prev_bpread).get_matching_blocks()
    for idx, block in enumerate(match_blocks):
        offset = block[1] - block[0]
        if offset in ns.keys():
            ns[offset] = ns[offset] + match_blocks[idx][2]
        else:
            ns[offset] = match_blocks[idx][2]
        nd[offset] = 0
#    for offset in range(-3,len(prev_bpread)):
#        pair = zip_longest(prev_bpread[offset:],bpread[:-offset],fillvalue=None)
#        comparison = [int(i==j) for i,j in pair]
#        ns[offset] = sum(comparison)
#        nd[offset] = len(comparison) - ns[offset]
    for key in ns.keys():
        if key < 0:
            k = -key
            log_px[key] = k*np.log((back_ratio)*N*jump_step_ratio) - sum([np.log(x+1) for x in range(k)]) +\
            ns[key]*np.log(p_same/0.25) + nd[key]*np.log(p_diff/0.25)
        else:
            log_px[key] = key*np.log(N*jump_step_ratio) - sum([np.log(x+1) for x in range(key)]) +\
            ns[key]*np.log(p_same/0.25) + nd[key]*np.log(p_diff/0.25)
    disp = max(log_px.keys(),key = lambda x: log_px[x])
    return disp,log_px[disp]

def global_alignment_kernal(bpread, prev_bpread):
    gap_open = -5
    gap_extend = -2
    mismatch = -3
    match = 1
    min_block_size = 3
    global_alignment = pairwise2.align.globalms(prev_bpread,bpread,match,mismatch,gap_open,gap_extend)
    if len(global_alignment) == 0:
        print(bpread)
        print(prev_bpread)
        raise ValueError("Alignment not found")
    blocks = match_blocks(global_alignment[0])
#    if criteria == 'first':
#        for block in blocks:
#            if block[0] >= min_block_size:
#                disp = block[1] - block[2]
#                break
#    elif criteria == "max":
    block = max(blocks, key = lambda x: x[0])
    disp = block[1] - block[2]
    if disp is None:
        disp = blocks[0][1] - blocks[0][2]
    return disp

def glue_kernal(bpread,prev_bpread):
    """
    This is a alignment for a larger jump step.
    A good setting would be jumpstep ~ 0.95 * segment_len
    """
    prev_n = len(prev_bpread)
    n = len(bpread)    
    max_overlap = min(math.floor(0.1 * prev_n),n)
    max_hit_disp = (0,0)
    for i in range(1,max_overlap):
        head=bpread[:i]
        tail=prev_bpread[-i:]
        head = np.asarray([x for x in head])
        tail = np.asarray([x for x in tail])
        score = 2*sum(head==tail) - i
        if score > max_hit_disp[1]:
            max_hit_disp = (i,score)
    disp = prev_n - max_hit_disp[0]
    return disp

def stick_kernal(bpread,prev_bpread):
    """
    Stick assembly,so basically it's just patch the reads together.
    """
    return(len(prev_bpread))

def simple_assembly(bpreads, jump_step_ratio, error_rate = 0.2,kernal = 'global',alphabeta = 'ACGT'):
    """
    Assemble the read from the chunks. Log probability is 
    Args:
        bpreads: Input chunks.
        jump_step_ratio: Jump step divided by segment length.
        error_rate: An estimating basecalling error rate.
        kernal: 'global': global alignment kernal, 'simple':simple assembly
    """
    base_n = len(alphabeta)
    concensus = np.zeros([base_n, 1000])
    pos = 0
    length = 0
    census_len = 1000
    for indx, bpread in enumerate(bpreads):
        if indx == 0:
            add_count(concensus, 0, bpread)
            continue
        prev_bpread = bpreads[indx - 1]
        if kernal == 'simple':
            disp,log_p = simple_assembly_kernal(bpread,prev_bpread,error_rate,jump_step_ratio)
        elif kernal == 'global':
            disp = global_alignment_kernal(bpread,prev_bpread,base_n)
        elif kernal == 'glue':
            disp = glue_kernal(bpread,prev_bpread)
        elif kernal == 'stick':
            disp = stick_kernal(bpread,prev_bpread)
        if disp + pos + len(bpreads[indx]) > census_len:
            concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                   mode='constant', constant_values=0)
            census_len += 1000
        add_count(concensus, pos + disp, bpreads[indx])
        pos += disp
        length = max(length, pos + len(bpreads[indx]))
    return concensus[:, :length]

def match_blocks(alignment):
    tmp_start = -1 
    blocks = []
    pos_0 = 0
    pos_1 = 0
    for idx,base in enumerate(alignment[0]):
        if (alignment[0][idx] == '-') or (alignment[1][idx] == '-'):
            if tmp_start >= 0:
                blocks.append([idx - tmp_start,pos_0,pos_1])
                tmp_start = -1
        else:
            if tmp_start == -1:
                tmp_start = idx
        if alignment[0][idx] != '-':
            pos_0 += 1
        if alignment[1][idx] != '-':
            pos_1 += 1
    if tmp_start >=0:
        blocks.append([idx - tmp_start,pos_0,pos_1])
    return blocks

def add_count(concensus, start_indx, segment):
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base][start_indx + i] += 1


###############################################################################

#########################Simple assembly method with quality score################################
def simple_assembly_qs(bpreads, qs_list, jump_step_ratio,error_rate = 0.2,kernal = 'global',alphabeta = 'ACGT'):
    """
    Assemble the read from the chunks. Log probability is 
    log_P ~ x*log((N*n1/L)) - log(x!) + Ns * log(P1/0.25) + Nd * log(P2/0.25)
    Args:
        bpreads: Input chunks.
        qs_list: Quality score logits list.
        jump_step_ratio: Jump step divided by segment length.
        error_rate: An estimating basecalling error rate.
        kernal: 'global': global alignment kernal, 'simple':simple assembly, 'glue':glue assembly, 'stick':stick assembly
    """
    base_n = len(alphabeta)
    concensus = np.zeros([base_n, 1000])
    concensus_qs = np.zeros([base_n, 1000])
    pos = 0
    length = 0
    census_len = 1000
    assert len(bpreads) == len(qs_list)
    for indx, bpread in enumerate(bpreads):
        if indx == 0:
            add_count_qs(concensus, concensus_qs, 0, bpread, qs_list[indx])
            continue
        prev_bpread = bpreads[indx - 1]
        if kernal == 'simple':
            disp,log_p = simple_assembly_kernal(bpread,prev_bpread,error_rate,jump_step_ratio)
        elif kernal == 'global':
            disp = global_alignment_kernal(bpread,prev_bpread)
        elif kernal == 'glue':
            disp = glue_kernal(bpread,prev_bpread)
        elif kernal == 'stick':
            disp = stick_kernal(bpread,prev_bpread)
        if disp + pos + len(bpread) > census_len:
            concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                   mode='constant', constant_values=0)
            concensus_qs = np.lib.pad(concensus_qs, ((0, 0), (0, 1000)),
                                      mode='constant', constant_values=0)
            census_len += 1000
        add_count_qs(concensus, concensus_qs, pos + disp, bpread, qs_list[indx])
        pos += disp
        length = max(length, pos + len(bpread))
    return concensus[:, :length], concensus_qs[:, :length]


def add_count_qs(concensus, concensus_qs, start_indx, segment, qs):
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base][start_indx + i] += 1
        concensus_qs[base][start_indx + i] += qs[i]


###############################################################################

def main():
    # bpreads = ['AAGGCCTAGCT','AGGCCTAGCAA','GGCCTAGCTC','AAAGGCCTAGT']
    #    logits_sample = np.load('/home/haotianteng/UQ/deepBNS/Chiron_Project/chiron_fastqoutput/chiron/utils/logits_sample.npy')
    start = time.time()
    # test = mc_path(logits_sample[300,:,:],base_type = 0)
    # print time.time()-start
#    bpreads = section_decoding(logits_sample)
# census = simple_assembly(bpreads)

#    result = np.argmax(census,axis=0)
#    print result
