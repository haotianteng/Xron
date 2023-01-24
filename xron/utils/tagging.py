#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 22:09:32 2022

@author: heavens
"""
import os
import re
import sys
import h5py
import numpy as np
import argparse
import pathlib
from tqdm import tqdm
from xron import __version__
FILE = pathlib.Path(__file__).resolve()
FLAGS = {"5mC":"C+m","5hmC":"C+h","5fC":"C+f","5caC":"C+c",
         "5hmU":"T+g","5fU":"T+e","5caU":"T+b",
         "6mA":"A+a",
         "8oxoG":"G+o",
         "Xao":"N+n"}
def check_index(fastq):
    if not os.path.exists(fastq+'.index'):
        raise FileNotFoundError('fastq index file not found')

def read_index(fastq):
    check_index(fastq)
    index_dict = {}
    with open(fastq+'.index') as f:
        index = f.read().splitlines()
        index_dict[index[0]] = index[1]
    return index_dict

def int8_encode(x):
    assert np.all(0 <= x) and np.all(x<= 1)
    return (x*256).astype(np.uint8)

class SamParser(object):
    def __init__(self,sam_file,fastq_index,modified,format='merge'):
        self.sam_file = sam_file
        self.fastq_index = fastq_index
        self.tagged = False #To identify if the file has been tagged by xron before.
        self.modified = modified
        self.flag = FLAGS[modified]
        self.canonical_base = self.flag.split('+')[0]
        self.format = format
    @property
    def PG_header(self):
        return '@PG\tID:xron_tagging\tPN:xron\tVN:%s\tCL:%s' % (__version__, 'python '+str(FILE)+' '+' '.join(sys.argv[1:]))

    def _remove_exisiting_xron_header(self):
        for header_line in self.headers:
            if 'ID:xron_tagging' in header_line:
                print("Found existing xron header, update it")
                self.headers.remove(header_line)
                self.tagged = True

    def add_header(self):
        self._remove_exisiting_xron_header()
        for i,h in enumerate(self.headers):
            #Edit the PP tag in the first PG header to create chain
            if h.startswith('@PG') and "PP:xron_tagging" not in h:
                if "PP:" in h:
                    raise ValueError("PP tag already exists in the first PG header")
                split_h = h.split('\t')
                h =  split_h[:2] + ['PP:xron_tagging'] + split_h[2:]
                h = '\t'.join(h)
                self.headers[i] = h
                break
        self.headers = [self.PG_header] + self.headers
    
    def read_fastq_index(self):
        read_ids, read_mappings = [],[]
        with open(self.fastq_index,'r') as f:
            for line in f:
                split_line = line.strip().split('\t')
                read_ids.append(split_line[0])
                read_mappings.append(split_line[1])
        self.read_ids = np.asarray(read_ids)
        self.read_mappings = np.asarray(read_mappings)

    def read_sam(self):
        with open(self.sam_file,'r') as f:
            sam = f.read().splitlines()
        self.headers,self.alignment = self._parse_header(sam)
        self.alignment_ids = [x.split('\t')[0] for x in self.alignment]
    
    def _parse_header(self,sam):
        return [line for line in sam if line.startswith('@')],[line for line in sam if not line.startswith('@')]

    def parse_MMflag(self,aln):
        split_aln = aln.strip().split('\t')
        MMs = {}
        MLs = {}
        for field in split_aln:
            if field.startswith('MM:Z:'):
                MM_tags = field[5:].strip(';').split(';')
                for MM_tag in MM_tags:
                    codes,positions = self.parse_MMtag(MM_tag)
                    for code,position in zip(codes,positions):
                        MMs[code] = position
            if field.startswith("ML:B:C"):
                split_ML = field[7:].split(',')
                assert len(split_ML) == sum(len(x.split(',')) for x in MMs.values())
                for key,val in MMs.items():
                    c = len(val.split(','))
                    MLs[key] = split_ML[:c]
                    del split_ML[:c]
        return MMs,MLs

    def parse_MMtag(self,MM_tag):
        #Parse single MM tag for each modification
        MM_tag_split = MM_tag.split(',')
        title = MM_tag_split[0].strip('.')
        unmo_base,mo_base = title.split('+')
        mo_codes,mo_pos = [],[]
        for m in mo_base:
            mo_codes.append(unmo_base+'+'+m)
            mo_pos.append(','.join(MM_tag_split[1:]))
        return mo_codes,mo_pos

    
    def _remove_exisiting_MMrecord(self,MMs,MLs):
        if self.flag in MMs.keys():
            del MMs[self.flag]
        if self.flag in MLs.keys():
            del MLs[self.flag]
    
    def _generate_MMtag(self,int8_modified_probability):
        #Generate MM tag for each modification
        if len(int8_modified_probability) == 0:
            return '',''
        pos = np.where(int8_modified_probability)[0]
        shift = pos[1:] - pos[:-1]
        shift -= 1
        if len(pos):
            gap_count = np.append([pos[0]],shift)
        else:
            gap_count = []
        mm = ','.join([str(x) for x in gap_count])
        ml = ','.join([str(x) for x in int8_modified_probability[pos]])
        return mm,ml
    
    def _generate_MMtag_nomerge(self,int8_modified_probability):
        #Generate MM tag with out merging 0 probability.
        if len(int8_modified_probability) == 0:
            return '',''
        mm = ','.join(['0']*len(int8_modified_probability))
        ml = ','.join([str(x) for x in int8_modified_probability])
        return mm,ml

    def read_modified(self,fast5_read_handle):
        modified_probability = None
        for entry in fast5_read_handle['Analyses'].keys():
            if not entry.startswith('Basecall_'):
                continue #Skip non-basecall entries
            if fast5_read_handle['Analyses'][entry].attrs['name'] != 'Xron':
                continue
            result_h = fast5_read_handle['Analyses'][entry]['BaseCalled_template']
            try:
                seq = str(np.asarray(result_h['Fastq']).astype(str)).split('\n')[1]
            except:
                seq = np.asarray(result_h['Fastq']).tobytes().decode('utf-8').split('\n')[1]
            try:
                modified_probability = np.asarray(result_h['ModifiedProbability'])
                if (len(modified_probability) == 0) and (seq.count(self.canonical_base)):
                    return None
                return int8_encode(modified_probability)
            except KeyError:
                continue
        return None

    def __call__(self):
        self.read_sam()
        self.read_fastq_index()
        self.add_header()
        fail_count = 0
        flag = self.flag
        mmtag_func = self._generate_MMtag if self.format == 'merge' else self._generate_MMtag_nomerge
        new_sam = self.headers
        uniq_file_list = set(self.read_mappings)
        with tqdm() as t:
            for fast5f in uniq_file_list:
             with h5py.File(fast5f,'r') as root:
                for read_id in self.read_ids[self.read_mappings==fast5f]:
                    try:
                        idx = self.alignment_ids.index(read_id)
                    except ValueError:
                        fail_count +=1
                        continue
                    aln = self.alignment[idx]
                    MMs,MLs = self.parse_MMflag(aln)
                    self._remove_exisiting_MMrecord(MMs, MLs)
                    read_id = aln.split('\t')[0]
                    t.postfix = "Read_id: %s, failed: %d"%(read_id,fail_count)
                    t.update()
                    read_h = root['read_' + read_id]
                    modified_p = self.read_modified(read_h)
                    if modified_p is None:
                        fail_count += 1
                    else:
                        mm,ml = mmtag_func(modified_p)
                        MMs[self.flag] = mm
                        MLs[self.flag] = ml
                    if len(MMs):
                        MM_string = 'MM:Z:'+';'.join([k+'.,'+v for k,v in MMs.items()])+';'
                        ML_string = 'ML:B:C,'+','.join([v for v in MLs.values()])
                        if 'MM:Z:' in aln:
                            aln = re.sub('MM:Z:.*?\t',MM_string+'\t',aln)
                        else:
                            aln = aln + '\t' + MM_string +'\t'
                        if 'ML:B:C' in aln:
                            aln = re.sub('ML:B:C.*?;',ML_string+';',aln)
                        else:
                            aln = aln + ML_string+';'
                        new_sam.append(aln)
        new_sam_file = os.path.splitext(self.sam_file)[0] + ".tagged.sam"
        with open(new_sam_file + '','w') as f:
            f.write('\n'.join(new_sam))

def main(args):
    if not args.fastq.endswith('.index'):
        args.fastq += '.index'
    if args.merge:
        format = "merge"
    else:
        format = "flatten"
    sam_writer = SamParser(args.sam,args.fastq,args.modified,format = format)
    sam_writer.read_sam()
    sam_writer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add modification tag into sam file.')
    parser.add_argument('--fastq', required = True, type=str, help='The merged fastq file')
    parser.add_argument('--sam', required = True, type=str, help='The sam file')
    parser.add_argument('--modified',default = "6mA", type=str, help='The modified base, \
                                                can be one of the %s'%list(FLAGS.keys()))
    parser.add_argument('--merge',action = "store_true",dest = "merge",
                        help = "Set the output MM tag format to compact format.")
    args = parser.parse_args()
    main(args)
