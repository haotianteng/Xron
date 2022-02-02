"""
Created on Mon Jun  7 23:59:58 2021

@author: Haotian Teng
"""

old_pore_f = "/home/heavens/twilight/CMU/Xron/xron/pore_models/control_hye.1.5mer_level_table.txt"
new_pore_f = "/home/heavens/twilight/CMU/Xron/xron/pore_models/5mer_level_table.model"
title = '\t'.join(['kmer','level_mean','level_stdv','sd_mean'])
with open(old_pore_f,'r') as f:
 with open(new_pore_f,'w+') as wf:
    wf.write(title+'\n')
    for line in f:
        split_line = line.strip().split()
        kmer = split_line[0]
        mean = split_line[2]
        std_line = next(f).strip().split()
        assert std_line[0] == kmer
        std = std_line[2]
        dwell_line = next(f).strip().split()
        assert dwell_line[0] == kmer
        dwell = dwell_line[2]
        wf.write('\t'.join([kmer,mean,std,dwell])+'\n')
        