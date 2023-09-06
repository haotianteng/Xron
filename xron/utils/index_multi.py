#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:47:48 2022
This script is slower than single thread index.py, probably because File IO is the bottleneck.
use index.py instead.
@author: heavens
"""
import os
import h5py
import argparse
from xron.utils.seq_op import fast5_shallow_iter
from xron.utils.fastIO import read_fastq
import multiprocessing as mp
from time import sleep 
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Queue
from tqdm import tqdm
import queue #In Python 2.7, this would be Queue

DESIRED_TIMEOUT = 1
MAX_result_dict_SIZE = 1000#This is the number of instances in the queue, the number should be set in the way that the max queue byte size smaller than the page capacity.
def worker(job_queue,fastqs,result_queue,log):
    while True:
        try:
            i = job_queue.get(timeout = DESIRED_TIMEOUT)
        except queue.Empty:
            return
        #Do the job here
        try:
            with h5py.File(i,mode = 'r') as root:
                read_ids = list(root.keys())
                for id in read_ids:
                    if id[5:] in fastqs['name']:
                        result_queue.put((id[5:],i))
            log['success'] += 1
        except Exception as e:
            print("Reading %s failed due to %s."%(i,e))
            log['fail'] += 1
            continue

def run(args):
    manager = Manager()  
    filequeue = manager.Queue()
    result_queue = manager.Queue()
    log = manager.dict()
    log['fail'] = 0
    log['success'] = 0
    
    max_threads_number = mp.cpu_count()-1 #1 thread is used for the main process
    all_proc = []
    fastq_records = read_fastq(args.fastq)
    fast5_records = {}
    file_number = 0
    print("Read in fast5 file list.")
    for (dirpath, dirnames, filenames) in os.walk(args.fast5+'/'):
        for filename in filenames:
            if not filename.endswith('fast5'):
                continue
            abs_path = os.path.join(dirpath,filename)
            filequeue.put(abs_path)
            file_number += 1
    print("Create read id to fast5 file mapping.")
    for i in range(max_threads_number if args.threads is None else args.threads):
        p = Process(target = worker, args = (filequeue,fastq_records,result_queue,log))
        all_proc.append(p)
        p.start()
    
    print("Indexing fastq files.")
    with open(args.fastq+'.index', 'w+') as f:
        with tqdm() as t:
            t.total = len(fastq_records['name'])
            while log['success'] + log['fail'] < file_number or not filequeue.empty() or not result_queue.empty():
                try:
                    #pop out result from the dict
                    result = result_queue.get()
                    if result[0] in fastq_records['name']:
                        f.write(result[0]+'\t'+result[1]+'\n')
                        fastq_records['name'].pop(fastq_records['name'].index(result[0]))
                        t.update()
                except KeyError:
                    sleep(0.1)
                    continue
    if len(fastq_records['name']) != 0:
        raise ValueError('%d fastq readid not found in fast5'%(len(fastq_records['name'])))       

    for p in all_proc:
        p.join()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fast5', required = True, type=str, help='folder that contains fast5 output')
    parser.add_argument('--fastq', required = True, type=str, help='The merged fastq file')
    parser.add_argument('--threads', type=int, default = None, help='The number of threads used for indexing')
    args = parser.parse_args()
    run(args)