"""
Created on Wed Apr 27 09:01:02 2022
This script is writen to load a HMM model and used it to summirize the output 
of a basecalled dataset.
@author: Haotian Teng
"""
import os
import time
import toml
import torch
import itertools
import numpy as np
from scipy.stats import binom
from tqdm import tqdm
from matplotlib import pyplot as plt
from xron.nrhmm.hmm import GaussianEmissions, RHMM
from xron.nrhmm.kmer2seq import fixing_looping_path
from xron.nrhmm.hmm_input import Kmer2Transition, Kmer_Dataset, Normalizer
from xron.utils.plot_op import auc_plot
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from xron.xron_train_base import DeviceDataLoader
from xron.utils.seq_op import Methylation_DP_Aligner
### Test module
from time import time

def kmers2seq(kmers,idx2kmer):
    merged = [g for g,_ in itertools.groupby(kmers)]
    seqs = [idx2kmer[x][0] for x in merged]
    return ''.join(seqs) + idx2kmer[merged[-1]][1:]

def print_pore_model(emission_means,idx2kmer,f = None):
    for i,mean in enumerate(emission_means):
        if f:
            f.write("%s: %.2f\n"%(idx2kmer[i],mean))
        else:
            print("%s: %.2f"%(idx2kmer[i],mean))
            
def load_pore_model(pore_model_f):
    pore_model = dict()
    with open(pore_model_f) as f:
        for line in f:
            split_line = line.split(":")
            pore_model[split_line[0]] = float(split_line[1])
    return pore_model

def multiple_testing(TP,FP,n = 100):
    TP_n = [1-binom.cdf(int(n/2),n,x) for x in TP]
    FP_n = [1-binom.cdf(int(n/2),n,x) for x in FP]
    return TP_n,FP_n

def posterior_decode(posterior,
                     idx2kmer,
                     kmer2idx,
                     AM_threshold:float = 1.):
    B,T,N = posterior.shape
    AM_ratio = []
    for p in posterior:
        path = torch.argmax(p,dim = 1)
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
        A_kmers = torch.tensor([kmer2idx[x] for x in A_kmers])
        M_kmers = torch.tensor([kmer2idx[x] for x in M_kmers])
        idxs = torch.tensor(idxs)
        A_p = p[idxs,A_kmers]
        M_p = p[idxs,M_kmers]
        cut = torch.log(A_p/M_p)<AM_threshold
        AM_ratio.append(torch.sum(cut).item()/(len(cut)))
    return AM_ratio
        

def compare_pore_models(pmf1:str, pmf2:str, selection:str):
    """
    Compare two pore models

    Parameters
    ----------
    pmf1 : str
        The string contains the file of the pore model.
    pmf2 : str
        The string contains the file of the second pore model.
    selection : str
        The magic string to filter the kmers to compare with, e.g. M!A, means
        most contain M and must not contain A in the kmer.

    Returns
    -------
    the difference of the kmers.
    """
    pm1 = load_pore_model(pmf1)
    pm2 = load_pore_model(pmf2)
    demand_char = [x for x in selection.split("!")[0]]
    filter_char = [x for x in selection.split("!")[1]]
    selected_kmers = [(val,pm2[key]) for key,val in pm1.items() if all([d in key for d in demand_char] + [f not in key for f in filter_char]) ]
    k1,k2 = list(zip(*selected_kmers))
    return np.asarray(k1),np.asarray(k2)

def run_test(args):
    model = torch.load(args.ckpt_f,
                       map_location = torch.device(args.device))
    # emission = GaussianEmissions(model['hmm']['emission.means'].cpu().numpy(), model['hmm']['emission.certainty'].cpu().numpy())
    emission = GaussianEmissions(model['hmm']['emission.means'].cpu().numpy(),model['hmm']['emission.cov'].cpu().numpy())
    print(model['hmm']['emission.cov'].cpu().numpy())
    hmm = RHMM(emission,normalize_transition=False,device = args.device)
    # sns.distplot(hmm.emission.certainty.cpu().numpy())
    # print(model['hmm']['emission.certainty'].cpu().numpy())
    config = toml.load(os.path.join(args.input,"config.toml"))
    print("Readout the pore model.")
    with open(args.ckpt_f+"_poremodel", "w+") as f:
        print_pore_model(model['hmm']['emission.means'],config['idx2kmer'],f)
    chunks = np.load(os.path.join(args.input,"chunks.npy"),mmap_mode="r")
    n_samples, sig_len = chunks.shape
    durations = np.load(os.path.join(args.input,"durations.npy"),mmap_mode = "r")
    idx2kmer = config['idx2kmer']
    kmer2idx = config['kmer2idx_dict']
    signal_collection = [[] for _ in np.arange(emission.N)]
    m6a_count = [0,0]
    thresholds = np.asarray(np.arange(-300,300,1)).astype(np.float)
    rocs = {x:[] for x in thresholds}
    mse,mse_renorm,m6a_ratios = [],[],[]
    kmers = np.load(os.path.join(args.input,"kmers.npy"))
    aligner = Methylation_DP_Aligner(base_alternation = {'M':'A'})
    k2t = Kmer2Transition(alphabeta = config['alphabeta'],
                          k = config['k'],
                          T_max = config['chunk_len'],
                          kmer2idx = config['kmer2idx_dict'],
                          idx2kmer = config['idx2kmer'],
                          neighbour_kmer = args.neighbour ,
                          base_alternation = {"M":"A"}, 
                          base_prior = {x:args.m_prior for x in config['alphabeta']},
                          kmer_replacement = True)
    dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
    loader = DataLoader(dataset,batch_size = args.batch_size, shuffle = False)
    loader = DeviceDataLoader(loader,device = args.device)
    norm = Normalizer(use_dwell = True,no_scale = True)
    count = 0
    for i_batch, batch in tqdm(enumerate(loader)):
        signal_batch = batch['signal']
        duration_batch = batch['duration']
        transition_batch = batch['labels']
        kmers_batch = batch['kmers']
        renorm_batch = signal_batch
        for j in tqdm(np.arange(args.n_renorm+1),desc = "Renorm the signal:"):
            with torch.no_grad():
                path,logit = hmm.viterbi_decode(renorm_batch, duration_batch, transition_batch)
            disagreement = torch.sum(path != kmers_batch,axis = 1)/duration_batch
            print("The disagreement between the doecded path and basecaller path: %.2f"%(torch.mean(disagreement)))
            rc_signal = np.asarray([[hmm.emission.means[x].item() for x in p] for p in path.cpu().numpy()])
            rc_signal_kmer = np.asarray([[hmm.emission.means[x].item() for x in p] for p in kmers_batch.cpu().numpy()])
            renorm_batch = norm(renorm_batch.cpu().numpy()[:,:,0],rc_signal, duration_batch.cpu().numpy(),path.cpu().numpy())
            renorm_batch = torch.from_numpy(renorm_batch).unsqueeze(dim = -1).to(args.device)
        renorm_batch = renorm_batch.squeeze(dim = -1).cpu().numpy()
        with torch.no_grad():
            gamma = hmm.expectation(torch.tensor(renorm_batch).float().unsqueeze(dim = -1).to(args.device), duration_batch, transition_batch)
            for t in tqdm(thresholds):
                rocs[t]+= posterior_decode(gamma,idx2kmer,kmer2idx,t)
        for i in np.arange(args.batch_size):
            if args.visual:
                fig,axs = plt.subplots()
                orig_sig = signal_batch[i][:duration_batch[i]].detach().cpu().numpy()
                rescale_sig = renorm_batch[i]
                axs.plot(orig_sig,c = 'g',label = "original signal")
                axs.plot(rescale_sig,c = 'r', label = "Rescaled signal")
                # rc_signal_orig = [hmm.emission.means[x].item() for x in kmers_batch[i]
                axs.plot(rc_signal[i][:duration_batch[i]],color = 'blue',label = 'reconstructed hmm',alpha = 0.7)
                axs.plot(rc_signal_kmer[i][:duration_batch[i]],color = 'yellow',label = 'reconstructed basecaller',alpha = 0.7)
                # axs.plot(rc_signal_orig[:duration_batch[i]],color = "red",label = "original label")
                diff = orig_sig[:,0] - rc_signal[i][:duration_batch[i]]
                diff_renorm = rescale_sig[:duration_batch[i]] - rc_signal[i][:duration_batch[i]]
                mse.append(np.mean(diff ** 2))
                mse_renorm.append(np.mean(diff_renorm ** 2))
                plt.legend()
                plt.xlabel("Time")
                plt.ylabel("Signal")
                plt.figure()
                dwell_length = np.asarray([len(list(x)) for _,x in itertools.groupby(path[i][:duration_batch[i]])])
                sns.histplot(dwell_length[dwell_length<10])
            deco_seq = kmers2seq(path.cpu().numpy()[i][:duration_batch[i]],idx2kmer)
            orig_seq = kmers2seq(kmers_batch.cpu().numpy()[i][:duration_batch[i]],idx2kmer)
            deco_aln,orig_aln = aligner.align(deco_seq,orig_seq)
            final_seq = aligner.merge(deco_aln,orig_aln)
            m6a_count[0] += final_seq.count('M')
            m6a_count[1] += final_seq.count('A')
            ratio = final_seq.count('M')/(final_seq.count('M')+final_seq.count('A'))
            m6a_ratios.append(ratio)
            print("m6A ratio %.2f"%(ratio))
            if args.visual:
                print("%d sample:"%(i))
                print("Decoded sequence:  %s"%(deco_seq))
                print("Align-corrected sequence: %s"%(final_seq))
                print("Original sequence: %s"%(orig_seq))
                print("MSE error: %.2f, MSE error renorm: %.2f"%(np.mean(mse),np.mean(mse_renorm)))
        count += 1
        if count >= args.n_cases:
            break
    m6a_ratios = np.asarray(m6a_ratios)
    print("m6A ratio %.2f"%(np.mean(m6a_ratios)))
    if args.visual:
        plt.plot(logit[0,np.arange(4000),kmers_batch[0]].detach().cpu().numpy(),label = "Oirignal.")
        plt.plot(logit[0,np.arange(4000),path[0]].detach().cpu().numpy(),label = "Realigned.")
        plt.ylabel("Log probability.")
        plt.xlabel("Time")
    return m6a_ratios,rocs

class Tester(object):
    def __init__(self,model_config):
        self.config = model_config
        self.datasets = {}
        self.loader = None
        self._load_model()
        
    def _load_model(self):
        model = torch.load(self.config.ckpt_f,
                           map_location = torch.device(self.config.device))
        emission = GaussianEmissions(model['hmm']['emission.means'].cpu().numpy(),model['hmm']['emission.cov'].cpu().numpy())
        self.hmm = RHMM(emission,normalize_transition=False,device = self.config.device)
    
    def _export_pore_model(self):
        with open(self.config.ckpt_f+"_poremodel", "w+") as f:
            print_pore_model(self.emission.means,self.data_config['idx2kmer'],f)
    
    def load_data_config(self,confiugre_file):
        config = toml.load(confiugre_file)
        self.data_config = config
        return config
    
    def add_dataset(self,dataset_folder,name = None):
        config = self.load_data_config(os.path.join(dataset_folder,"config.toml"))
        chunks = np.load(os.path.join(dataset_folder,"chunks.npy"),mmap_mode="r")
        n_samples, sig_len = chunks.shape
        durations = np.load(os.path.join(dataset_folder,"durations.npy"),mmap_mode = "r")
        kmers = np.load(os.path.join(dataset_folder,"kmers.npy"),mmap_mode = "r")
        k2t = Kmer2Transition(alphabeta = self.data_config['alphabeta'],
                          k = config['k'],
                          T_max = config['chunk_len'],
                          kmer2idx = config['kmer2idx_dict'],
                          idx2kmer = config['idx2kmer'],
                          neighbour_kmer = self.config.neighbour,
                          base_alternation = {"M":"A"}, 
                          base_prior = {"M":self.config.m_prior},
                          kmer_replacement = True)
        dataset = Kmer_Dataset(chunks, durations, kmers,transform=transforms.Compose([k2t]))
        name = name if name is not None else "dataset %d"%(len(self.datasets.keys())+1)
        self.datasets[name] = dataset

    def load_dataset(self,name,shuffle = True):
        if len(self.datasets.keys()) == 0:
            raise ValueError("No avaliable dataset is found, please add dataset first.")
        dataset = self.datasets[name]
        loader = DataLoader(dataset,batch_size = self.config.batch_size, shuffle = shuffle)
        self.loader = DeviceDataLoader(loader,device = self.config.device)
        return loader
    
    def renormalization(self, 
                        batch,
                        n_renorm = None):
        if n_renorm is None:
            n_renorm = self.config.n_renorm
        norm = Normalizer(use_dwell = self.config.dwell_norm,
                          no_scale = self.config.no_scale)
        signal_batch = batch['signal']
        duration_batch = batch['duration']
        transition_batch = batch['labels']
        kmers_batch = batch['kmers']
        renorm_batch = signal_batch
        for j in tqdm(np.arange(n_renorm),desc = "Renorm the signal:"):
            with torch.no_grad():
                path,logit = self.hmm.viterbi_decode(renorm_batch, duration_batch, transition_batch)
                for i in np.arange(len(path)):
                    seq = kmers2seq(kmers_batch[i],idx2kmer = self.data_config['idx2kmer'])
                    if config.fix_alignment:
                        p = path[i].cpu().numpy()
                        d = duration_batch[i].cpu().numpy()
                        fix_path = fixing_looping_path(p[:d],seq,idx2kmer = self.data_config['idx2kmer'],canonical_base = 'A',modified_base = 'M')
                        path[i,:duration_batch[i]] = torch.from_numpy(fix_path)
            rc_signal = np.asarray([[self.hmm.emission.means[x].item() for x in p] for p in path.cpu().numpy()])
            rc_signal_kmer = np.asarray([[self.hmm.emission.means[x].item() for x in p] for p in kmers_batch.cpu().numpy()])
            renorm_batch = norm(renorm_batch.cpu().numpy()[:,:,0],rc_signal, duration_batch.cpu().numpy(),path.cpu().numpy())
            renorm_batch = torch.from_numpy(renorm_batch).unsqueeze(dim = -1).to(self.config.device)
        return renorm_batch,rc_signal,rc_signal_kmer
    
    def renormalization_test(self,
                             n_batch = 1,
                             plot = False):
        if self.loader is None:
            raise ValueError("No dataset has been loaded, call Tester.load_dataset first.")
        collection = {"RC_MSE":[],"RC_RENORM_MSE":[]}
        for i_batch, batch in tqdm(enumerate(self.loader)):
            if i_batch >= n_batch:
                break
            signal_batch = batch['signal']
            duration_batch = batch['duration']
            transition_batch = batch['labels']
            kmers_batch = batch['kmers']
            renorm_batch,rc_signal,rc_signal_kmer = self.renormalization(batch)
            with torch.no_grad():
                path,logit = self.hmm.viterbi_decode(renorm_batch, duration_batch, transition_batch)
            renorm_batch = renorm_batch.squeeze(dim = -1).cpu().numpy()
            for i in np.arange(self.config.batch_size):
                orig_sig = signal_batch[i][:duration_batch[i]].detach().cpu().numpy()
                rescale_sig = renorm_batch[i]
                diff = orig_sig[:,0] - rc_signal[i][:duration_batch[i]]
                diff_renorm = rescale_sig[:duration_batch[i]] - rc_signal[i][:duration_batch[i]]
                collection['RC_MSE'].append(np.mean(diff ** 2))
                collection['RC_RENORM_MSE'].append(np.mean(diff_renorm ** 2))
                if plot:
                    print("Sequence %d"%(i))
                    if self.config.fix_alignment:
                        seq = kmers2seq(kmers_batch[i],idx2kmer = self.data_config['idx2kmer'])
                        fix_path = fixing_looping_path(path[i,:duration_batch[i]],seq,idx2kmer = self.data_config['idx2kmer'],canonical_base = 'A',modified_base = 'M')
                        print("pred:%s"%(kmers2seq(fix_path,self.data_config['idx2kmer'])))
                    else:
                        print("pred:%s"%(kmers2seq(path[i],self.data_config['idx2kmer'])))
                    print("orig:%s"%(kmers2seq(kmers_batch[i],self.data_config['idx2kmer'])))
                    fig,axs = plt.subplots()
                    axs.plot(orig_sig,c = 'g',label = "original signal")
                    axs.plot(rescale_sig,c = 'r', label = "Rescaled signal")
                    # rc_signal_orig = [hmm.emission.means[x].item() for x in kmers_batch[i]
                    axs.plot(rc_signal[i][:duration_batch[i]],color = 'blue',label = 'reconstructed hmm',alpha = 0.7)
                    axs.plot(rc_signal_kmer[i][:duration_batch[i]],color = 'yellow',label = 'reconstructed basecaller',alpha = 0.7)
                    # axs.plot(rc_signal_orig[:duration_batch[i]],color = "red",label = "original label")
                    plt.legend()
                    plt.xlabel("Time")
                    plt.ylabel("Signal")
                    plt.figure()
                    dwell_length = np.asarray([len(list(x)) for _,x in itertools.groupby(path[i][:duration_batch[i]])])
                    sns.histplot(dwell_length[dwell_length<10])
        return collection
            
    def proportion_test(self,
                        n_batch = 1):
        if self.loader is None:
            raise ValueError("No dataset has been loaded, call Tester.load_dataset first.")
        collection = {"m_prop":[],
                      "m_count":[],
                      "a_count":[],
                      "m_prop_aln":[],
                      "a_count_aln":[],
                      "m_count_aln":[]}
        for i_batch, batch in tqdm(enumerate(self.loader)):
            if i_batch >= n_batch:
                break
            duration_batch = batch['duration']
            transition_batch = batch['labels']
            kmers_batch = batch['kmers']
            renorm_batch,_,_ = self.renormalization(batch)
            with torch.no_grad():
                path,logit = self.hmm.viterbi_decode(renorm_batch, duration_batch, transition_batch)
            for i in np.arange(self.config.batch_size):
                deco_seq = kmers2seq(path.cpu().numpy()[i][:duration_batch[i]],self.data_config['idx2kmer'])
                orig_seq = kmers2seq(kmers_batch.cpu().numpy()[i][:duration_batch[i]],self.data_config['idx2kmer'])
                aligner = Methylation_DP_Aligner(base_alternation = {'M':'A'})
                deco_aln,orig_aln = aligner.align(deco_seq,orig_seq)
                final_seq = aligner.merge(deco_aln,orig_aln)
                collection["m_count"].append(deco_seq.count("M"))
                collection["a_count"].append(deco_seq.count("A"))
                collection["m_prop"].append(deco_seq.count('M')/(deco_seq.count('A')+deco_seq.count('M')))
                collection["m_count_aln"].append(final_seq.count('M'))
                collection["a_count_aln"].append(final_seq.count('A'))
                collection["m_prop_aln"].append(final_seq.count('M')/(final_seq.count('M')+final_seq.count('A')))
        return collection
    
    def auc_test(self,
                 n_batch = 1,
                 thresholds = []):
        if self.loader is None:
            raise ValueError("No dataset has been loaded, call Tester.load_dataset first.")
        collection = {"TP":[],"FP":[]}
        for i_batch, batch in tqdm(enumerate(self.loader)):
            if i_batch >= n_batch:
                break
            signal_batch = batch['signal']
            duration_batch = batch['duration']
            transition_batch = batch['labels']
            kmers_batch = batch['kmers']
            renorm_batch,rc_signal,rc_signal_kmer = self.renormalization(batch)
            rocs = {x:[] for x in thresholds}
            with torch.no_grad():
                gamma = self.hmm.expectation(renorm_batch.float(), duration_batch, transition_batch)
            for t in tqdm(thresholds):
                rocs[t]+= posterior_decode(gamma,self.data_config['idx2kmer'],self.data_config['kmer2idx_dict'],t)
        return rocs
    
if __name__ == "__main__":
    import seaborn as sns
    torch.manual_seed(1992)
    home_f = os.path.expanduser('~') 
    class ModelArguments:
        batch_size = 10
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m_prior = 0.3 #The prior factor of a base being normalized, 1 means no bias.
        neighbour = 4
        ckpt_f = None
        ## Renormalization arguments
        dwell_norm = True
        no_scale = True
        n_renorm = 3
        fix_alignmetn = False
        
        ## arguments for old function
        visual = False
        n_cases = 1
        
    model_config = ModelArguments
    
    ### Models
    # ckpts = ['/home/heavens/bridge_scratch/NRHMM_models/xron_rhmm_models_new/ckpt-36234',
    #          '/home/heavens/bridge_scratch/NRHMM_models/xron_rhmm_models_new/ckpt-31613',
    #          '/home/heavens/bridge_scratch/NRHMM_models/xron_rhmm_models_new/ckpt-12855']
    ckpts = ['/home/heavens/bridge_scratch/NRHMM_models/rhmm_mm_norm/ckpt-8609']
    compare = {"m6A":home_f + "/bridge_scratch/ime4_Yearst/IVT/m6A/rep2/kmers_guppy_4000_noise",
                "control":home_f + "/bridge_scratch/NA12878_RNA_IVT/xron_output/kmers_xron_4000_noise/"}
    
    # compare = {"m6A":home_f + "/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_25_pct/20210430_1751_X1_FAQ16555_e98b69f8/kmers_guppy_4000_noise"}
    # compare = [home_f + "/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/kmers_guppy_4000_dwell",
    #             "/home/heavens/bridge_scratch/NA12878_RNA_IVT/xron_partial/extracted_kmers"]
    # compare = [home_f + "/bridge_scratch/NA12878_RNA_IVT/guppy_train/kmers_guppy_4000_dwell"]
    # compare = ["/home/heavens/bridge_scratch/NA12878_RNA_IVT/xron_partial/extracted_kmers"]
    # compare = ["/home/heavens/bridge_scratch/NA12878_RNA_IVT/xron_partial/extracted_kmers"]
    # compare = [home_f + "/bridge_scratch/ime4_Yearst/IVT/m6A/rep1/kmers_guppy_4000_dwell"]
    
    testers = []
    for ckpt in ckpts:
        model_config.ckpt_f = ckpt
        testers.append(Tester(model_config))
    
    # testers[0].add_dataset(compare["m6A"],name="P25")
    print("Adding datasets.")
    testers[0].add_dataset("/home/heavens/bridge_scratch/m6A_Nanopore_RNA002/data/m6A_25_pct/20210430_1751_X1_FAQ16555_e98b69f8/kmers_guppy_4000_noise",name = "P25")
    testers[0].add_dataset(home_f + "/bridge_scratch/NA12878_RNA_IVT/xron_output/kmers_xron_4000_noise/",name = "control")
    testers[0].add_dataset(home_f + "/bridge_scratch/ime4_Yearst/IVT/m6A/rep2/kmers_guppy_4000_noise",name = "m6Arep1")
    
    print("Datasets added successfully, loading the control dataset.")
    testers[0].load_dataset("control")
    
    print("Renormalization test.")
    summary = testers[0].renormalization_test(n_batch = 1,plot = True)
    
    print("Calculate ROC on control dataset.")
    roc_control = testers[0].auc_test(n_batch = 2, thresholds = np.arange(-100,100,0.5))
    
    print("Loading m6A dataset.")
    testers[0].load_dataset("m6Arep1")
    
    print("Calculate ROC on m6A dataset.")
    roc_meth = testers[0].auc_test(n_batch = 2, thresholds = np.arange(-100,100,0.5))
    
    # summary = testers[0].proportion_test(n_batch = 10)
    testers[0].load_dataset("m6Arep1")
    # summary_m6A = testers[0].proportion_test(n_batch = 10)
    TP = [np.mean(x) for x in roc_meth.values()]
    FP = [np.mean(x) for x in roc_control.values()]
    fig,axs = plt.subplots(figsize = (5,5))
    auc_plot(TP,FP,axs)
    fig,axs = plt.subplots(figsize = (5,5))
    TP_n,FP_n = multiple_testing(TP, FP, 10)
    auc_plot(TP_n,FP_n,axs)
    
