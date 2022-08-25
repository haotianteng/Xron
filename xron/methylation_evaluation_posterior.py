"""
Created on Wed Jul 20 11:04:29 2022

@author: Haotian Teng
"""
import numpy as np
from xron.utils.seq_op import fast5_iter
from fast_ctc_decode import beam_search
# seq,path = beam_search(p,
#                        "N"+self.config.CTC["alphabeta"],
#                        beam_size = self.config.CTC["beam"],
#                        beam_cut_threshold = self.config.CTC["beam_cut_threshold"])
def AUC(TP,FP):
    """Calculate Area under curve given the true positive and false positive
    array
    """
    TP = TP[::-1] if TP[0]>TP[-1] else TP
    FP = FP[::-1] if FP[0]>FP[-1] else FP
    TP = [0] + TP if TP[0] != 0 else TP
    TP = TP + [1] if TP[-1] != 1 else TP
    FP = [0] + FP if FP[0] != 0 else FP
    FP = FP + [1] if FP[-1] != 1 else FP
    FP = np.asarray(FP)
    TP = np.asarray(TP)
    return np.sum((TP[1:] + TP[:-1])*(FP[1:]-FP[:-1])/2)
    
def posterior_decode(posterior,
                     idx2kmer = ['A','C','G','T','M','b'],
                     kmer2idx = {'A':0,'C':1,'G':2,'T':3,'M':4,'b':5},
                     AM_threshold:float = 1.):
    B,T,N = posterior.shape
    AM_ratio = []
    for p in posterior:
        path = np.argmax(p,axis = 1)
        kmers = [idx2kmer[x.item()] for x in path]
        A_kmers,M_kmers = [],[]
        idxs = []
        for i,kmer in enumerate(kmers):
            if kmer[2] == 'A' or kmer[2] == 'M':
                kmer = ''.join([x if i!=2 else 'A' for i,x in enumerate(list(kmer)) ])
                A_kmers.append(kmer)
                kmer = ''.join([x if i!=2 else 'M' for i,x in enumerate(list(kmer)) ])
                M_kmers.append(kmer)
                idxs.append(i)
        A_kmers = np.asarray([kmer2idx[x] for x in A_kmers])
        M_kmers = np.asarray([kmer2idx[x] for x in M_kmers])
        idxs = np.asarray(idxs)
        A_p = p[idxs,A_kmers]
        M_p = p[idxs,M_kmers]
        cut = (A_p/M_p)<AM_threshold
        AM_ratio.append(np.sum(cut)/(len(cut)))
    return AM_ratio

def read_fast5(read_h,index = "000"):
    result_h = read_h['Analyses/Basecall_1D_%s/BaseCalled_template'%(index)]
    logits = result_h['Logits']
    move = result_h['Move']
    try:
        seq = np.asarray(result_h['Fastq']).tobytes().decode('utf-8').split('\n')[1]
    except:
        seq = str(np.asarray(result_h['Fastq']).astype(str)).split('\n')[1]
    return np.asarray(logits),np.asarray(move),seq

def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits),axis = 1,keepdims = True)
    

if __name__ == "__main__":
    for read_h,signal,abs_path,read_id in fast5_iter("/home/heavens/Documents/test_fast5s"):
        logits,move,seq = read_fast5(read_h,index = "001")
        break