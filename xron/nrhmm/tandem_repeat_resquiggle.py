"""
This script is used to resquiggle the reads that have different tandem repeat number than the reference.
"""
import os
import toml
import argparse
import numpy as np
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Queue
from xron.nrhmm.kmer2seq import fixing_looping_path
import multiprocessing as mp
from time import sleep 
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Queue
from tqdm import tqdm
import queue #In Python 2.7, this would be Queue

DESIRED_TIMEOUT = 1
MAX_RESULT_QUEUE_SIZE = 1000
SLEEPING_TIME = 0.01
def worker(job_queue,args,result_queue):
    while True:
        try:
            i,seq,path,duration = job_queue.get(timeout = DESIRED_TIMEOUT)
        except queue.Empty:
            return
        #Do the job here
        if len(seq) < args.min_seq_len:
            result_queue.put({i:(None,None)})
        else:
            fixed_path, fixing = fixing_looping_path_func(path[:duration],seq)
            # while result_queue.qsize() > MAX_RESULT_QUEUE_SIZE:
            #     sleep(SLEEPING_TIME)
            result_queue.put({i:(fixed_path,fixing)})

def main(args):
    seqs,path,durations,config = load_data(args.input_folder)
    manager = Manager()
    filequeue = manager.Queue()
    result_queue = manager.Queue(maxsize = MAX_RESULT_QUEUE_SIZE)
    results_dict = {"Fixed":0, "No need to fix":0, "Fix failed":0}
    max_threads_number = mp.cpu_count()-1 if args.threads is None else args.threads #1 thread is used for the main process
    all_proc = []
    file_number = len(path)
    for i in range(file_number):
        filequeue.put((i,seqs[i],path[i],durations[i]))
    for i in range(max_threads_number):
        p = Process(target = worker, args = (filequeue,args,result_queue))
        all_proc.append(p)
        p.start()
    to_finished = list(np.arange(file_number))
    with tqdm() as t:   
     t.total = file_number
     t.set_description("Fixing the path of control data.")
     while len(to_finished):
        try:
            result = result_queue.get(timeout = DESIRED_TIMEOUT)
            i = list(result.keys())[0]
            fixed_path, fixing = result[i]
            if fixing is None:
                results_dict["Fix failed"] += 1
            elif fixing:
                results_dict['Fixed'] += 1
                path[i][:durations[i]] = fixed_path
            else:
                results_dict['No need to fix'] += 1
            elements_in_queue = result_queue.qsize()
            t.set_description(f"Fixed {results_dict['Fixed']}, No need to fix {results_dict['No need to fix']}, Skip because sequence is too short {results_dict['Fix failed']}, elements in queue {elements_in_queue}")
            if i in to_finished:
                to_finished.remove(i)
            else:
                raise ValueError(f"The index {i} is not in the to_finished list.")
            t.update()
        except queue.Empty:
            continue
        except Exception as e:
            print(e)
            break
    print("Saving the delooped result.")
    np.save(os.path.join(args.input_folder,"path_fix.npy"),path)
    print("Finished")
    for i,p in enumerate(all_proc):
        p.join()

def load_data(input_folder):
    seqs = np.load(os.path.join(input_folder,"seqs.npy"))
    path = np.load(os.path.join(input_folder,"path.npy"))
    duration = np.load(os.path.join(input_folder,"durations.npy"))
    config = toml.load(os.path.join(input_folder,"config.toml"))
    return seqs,path,duration,config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_folder",help = "The folder that contains the data to be fixed.")
    parser.add_argument("--modified_base",default = "M",help = "The modified base.")
    parser.add_argument("--canonical_base",default = "A",help = "The canonical base.")
    parser.add_argument("--threads",default = None,type = int,help = "The number of threads to use, default is None which means using all the threads.")
    parser.add_argument("--min_seq_len",default = 7,type = int,help = "The minimum length of the sequence to be fixed.")
    args = parser.parse_args()
    config = toml.load(os.path.join(args.input_folder,"config.toml"))
    fixing_looping_path_func = partial(fixing_looping_path,idx2kmer = config['idx2kmer'],modified_base=args.modified_base,canonical_base=args.canonical_base)
    main(args)