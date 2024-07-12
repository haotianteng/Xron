#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:35:26 2022

@author: heavens
"""
import os
import numpy as np
import lmdb
import shelve
import pickle
import zlib
from abc import ABC, abstractmethod

class Database(ABC):
    def __init__(self, compress = True):
        self.compress = compress
        self.db = None

    @abstractmethod
    def load(self,index_f):
        pass

    @abstractmethod
    def get_item(self,key):
        pass

    @abstractmethod
    def get_keys(self):
        pass

    @abstractmethod
    def get_values(self):
        pass

    @abstractmethod
    def get_items(self):
        pass

class ShelveDatabase(Database):
    def get_item(self,key):
        content = self.db[key]
        if self.compress:
            return pickle.loads(zlib.decompress(content))
        else:
            return pickle.loads(content)
        
    def load(self,index_f):
        self.db = shelve.open(index_f)

    def get_keys(self):
        return self.db.keys()
    
    def get_values(self):
        return self.db.values()
    
    def get_items(self):
        return self.db.items()
    
    def __del__(self):
        if self.db is not None:
            self.db.close()

class LmdbDatabase(Database):
    def get_item(self,key):
        if isinstance(key,str):
            key = key.encode('utf-8')
        content = self.db.get(key)
        if content is None:
            raise KeyError(f"{key}")
        if self.compress:
            return pickle.loads(zlib.decompress(content))
        else:
            return pickle.loads(content)
    
    def load(self,index_f):
        self.env = lmdb.open(index_f, readonly=True)
        self.db = self.env.begin()
    
    def get_keys(self):
        yield from self.db.cursor().iternext(keys=True,values=False)

    def get_values(self):
        yield from self.db.cursor().iternext(keys=False,values=True)
    
    def get_items(self):
        yield from self.db.cursor().iternext(keys=True,values=True)
    
    def __del__(self):
        if self.env is not None:
            self.env.close()

def get_db_instance(backend = "shelve", compress = True):
    if backend == "shelve":
        return ShelveDatabase(compress=compress)
    elif backend == "lmdb":
        return LmdbDatabase(compress=compress)
    else:
        raise ValueError("Invalid backend")

class Indexer(dict):
    def __init__(self, 
                 backend = "shelve",
                 compress = True):
        super().__init__()
        self.backend = get_db_instance(backend,compress)

    def load(self,index_f):
        self.backend.load(index_f)

    def __getitem__(self,key):
        return self.backend.get_item(key)        
    
    def __iter__(self):
        for key,val in self.backend.get_items():
            #decode key if it is bytes
            if isinstance(key,bytes):
                key = key.decode('utf-8')
            if self.backend.compress:
                val = pickle.loads(zlib.decompress(val))
            yield key,val

    def __repr__(self):
        return f"Indexer({self.backend})"

    def __str__(self):
        return f"Indexer({self.backend})"
    
    def __del__(self):
        self.backend.__del__()

def read_fastqs(fastq_f):
    records = {"sequences":[],"name":[],"quality":[]}
    for fastq in os.listdir(fastq_f):
        with open(os.path.join(fastq_f,fastq),'r') as f:
            for line in f:
                if line.startswith("@"):
                    records['name'].append(line.strip()[1:])
                    records['sequences'].append(next(f).strip())
                    assert next(f).strip() == "+" #skip the "+"
                    records['quality'].append(next(f).strip())
    return records

def read_fastq(fastq):
    records = {"sequences":[],"name":[],"quality":[]}
    with open(fastq,'r') as f:
        for line in f:
            if line.startswith("@"):
                records['name'].append(line.strip()[1:])
                records['sequences'].append(next(f).strip())
                assert next(f).strip() == "+",print(line) #skip the "+"
                records['quality'].append(next(f).strip())
    return records

def read_fast5(read_h,index = "000"):
    result_h = read_h['Analyses/Basecall_1D_%s/BaseCalled_template'%(index)]
    logits = result_h['Logits']
    move = result_h['Move']
    try:
        seq = str(np.asarray(result_h['Fastq']).astype(str)).split('\n')[1]
    except:
        seq = np.asarray(result_h['Fastq']).tobytes().decode('utf-8').split('\n')[1]
    return np.asarray(logits),np.asarray(move),seq

def read_entry(read_h,entry:str,index = "000"):
    """
    Read a entry given the name

    """
    result_h = read_h['Analyses/Basecall_1D_%s/BaseCalled_template'%(index)]
    return np.asarray(result_h[entry])

if __name__ == "__main__":
    lmdb_index = "/data/HEK293T_RNA004/aligned.sorted.bam.index"
    shelve_index = "/data/HEK293T_RNA004/index_test/aligned.sorted.bam.index"
    
    #test lmdb database
    indexer_lmdb = Indexer(backend = "lmdb")
    indexer_lmdb.load(lmdb_index)
    for key,val in indexer_lmdb:
        print(key,val)
        break
    try:
        indexer_lmdb['aaa'] #should rase KeyError
    except KeyError as e:
        print(e)
    indexer_lmdb['00000097-e849-4535-b270-fa6dd6a2ec83']

    #test shelve database
    indexer_shelve = Indexer(backend = "shelve")
    indexer_shelve.load(shelve_index)
    for key,val in indexer_shelve:
        print(key,val)
        break
    try:
        indexer_shelve['aaa'] #should rase KeyError
    except KeyError as e:
        print(e)
    indexer_shelve['00000097-e849-4535-b270-fa6dd6a2ec83']