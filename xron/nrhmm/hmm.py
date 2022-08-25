"""
Created on Wed Mar 30 12:13:39 2022

@author: Haotian Teng
"""
import torch
import itertools
from torch.nn.functional import normalize,one_hot
import numpy as np
from typing import List,Union
class GaussianEmissions(torch.nn.Module):
    def __init__(self, means:np.array,cov:np.array, trainable:np.array = None, fix_cov = True):
        """
        The Gaussian emission module to generate .

        Parameters
        ----------
        means : np.array, shape [N,K]
             The means of the N states, K is the number of features.
        cov : np.array, shape [N,K]
            The diagonal covariance matrix of the N states and K feature.
        trainable : np.array, [N]
            To mark if the state is trainable. The default is None, will train
            all the states.
        fix_cov : bool
            If we want to fix the covariance during training.

        """
        super().__init__()
        self.N, self.K = means.shape
        self.means = torch.nn.Parameter(torch.from_numpy(means).float())
        self.trainable = torch.from_numpy(trainable).bool() if trainable is not None else torch.Tensor([1]*self.N).bool()
        self.cov = torch.nn.Parameter(torch.from_numpy(cov).float(),requires_grad = not fix_cov)
        self.certainty = torch.nn.Parameter(torch.zeros((self.N,self.K)),requires_grad = False)
        self.momentum_mean = torch.nn.Parameter(torch.zeros((self.N,self.K)),requires_grad = False)
        self.momentum_mean_square = torch.nn.Parameter(torch.zeros((self.N,self.K)),requires_grad = False)
        self.global_step = torch.nn.Parameter(torch.tensor([1]*self.N),requires_grad = False)
        self.cache = None #The cache when Baum-Welch is used.
        self.epsilon = 1e-6 # A small number to prevent numerical instability
        self.fix_cov = fix_cov
        if self.trainable is not None:
            self.means.register_hook(lambda grad: grad*self.trainable[:,None].to(self.device))
    
    def reinitialize_means(self, means:torch.tensor):
        """
        Reinitialize the means, this function is used to update the means of
        the modified kmers.

        Parameters
        ----------
        means : torch.tensor
            The new means.
        """
        with torch.no_grad():
            self.means.copy_(means)
            if hasattr(self,"global_step"):
                self.global_step = 1
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Get the emission probability given a observation at time t.

        Parameters
        ----------
        x : torch.Tensor with shape [*,K]
            The observation value at certain time point, B is the batch size
            and K is the number of feature.

        Returns
        -------
        posterior: torch.Tensor with shape [*,N], return the log posterior 
        probability P(z|x), N is the number of states.
        """
        return -0.5*((1/self.cov*(self.means-x.unsqueeze(dim = -2))**2).sum(dim = -1)+torch.log(self.cov).sum(dim = 1)+self.K*np.log(2*np.pi))
    
    @torch.no_grad()
    def clamp_cov(self):
        """
        Clamp the covariance to positive number

        """
        self.cov[:] = torch.clamp(self.cov,min = self.epsilon)
    
    @torch.no_grad()        
    def update_cov(self,
                   method = "max",
                   exploration = 1.):
        """
        Set the uniform covariance according to the uncertainty, this function
        is useful when the covariance need to be reset during training.

        """
        if method == "max":
            cov = torch.max(self.certainty[self.trainable])
        else:
            raise NotImplementedError("Other covariance reestimation methods \
                                      beside max is not implemented .")
        print("Rescale the covariance to %.2f"%(cov))
        self.cov[:] = cov*exploration
    
    def _initialize_sufficient_statistics(self):
        self.cache = {"means":torch.zeros((self.N,self.K),device = self.device),
                      "covs":torch.zeros((self.N,self.K),device = self.device),
                      "posterior":torch.zeros(self.N,device = self.device)}
        
    @torch.no_grad()
    def _accumulate_sufficient_statistics(self, x:torch.Tensor, gamma:torch.Tensor):
        """
        Accumulate the sufficient statistics for the emission model given batch
        of the observation and posterior probability of the hidden variable.

        Parameters
        ----------
        x : torch.Tensor with shape [B,T,K]
            A batch of the observation time-series signal.
        gamma : torch.Tensor with shape [B,T,N]
            The posterior probability of the given observation batch.
        """
        if self.cache is None:
            self._initialize_sufficient_statistics()
        self.cache["means"] += (x.unsqueeze(dim = 2)*gamma.unsqueeze(dim = 3)).sum(dim = (0,1))
        self.cache["covs"] += (((x.unsqueeze(dim = 2)-self.means)**2)*gamma.unsqueeze(dim = 3)).sum(dim = (0,1))
        self.cache["posterior"] += gamma.sum(dim = (0,1))
    
    @torch.no_grad()
    def update_parameters(self, 
                          lr:float = 0.1, 
                          weight_decay:float = 0.0,
                          momentum:float = 0.9,
                          second_momentum:float = 0.999,
                          eps = 1e-8):
        """
        Update the parameter from the accumulated sufficient statistics, this 
        method was used if the model is updated using the Baum-Welch algorithm
        instead of direct gradient descent.
        
        Parameters
        ----------
        moving_average: float, optional
            The factor a of the moving average, the parameter will be updated 
            by aP_{t-1} + (1-a)P_t, default is 0 which means no smoothing will
            be used.

        """
        g_mean = self.means - self.cache["means"]/(self.cache["posterior"]+self.epsilon).unsqueeze(dim = 1)
        self.trainable = self.trainable.to(self.device)
        update_mask = torch.logical_and(self.trainable,self.cache["posterior"]>0)
        not_update_mask = torch.logical_not(update_mask)
        if weight_decay > 0:
            g_mean += weight_decay * self.means
        if momentum != 0:
            if not hasattr(self,"momentum_mean"):
                self.momentum_mean = torch.nn.Parameter(torch.zeros((self.N,self.K),device = self.device),requires_grad = False)
                self.momentum_mean_square = torch.nn.Parameter(torch.zeros((self.N,self.K),device = self.device),requires_grad = False)
                self.global_step = torch.nn.Parameter(torch.tensor([1]*self.N,device = self.device),requires_grad = False)
            new_momentum = momentum*self.momentum_mean + (1-momentum)*g_mean
            new_second_momentum = second_momentum*self.momentum_mean_square + (1-second_momentum)*(g_mean**2)
            # new_momentum[not_update_mask] = self.momentum_mean[not_update_mask]
            # new_second_momentum[not_update_mask] = self.momentum_mean_square[not_update_mask]
            self.momentum_mean[update_mask] = new_momentum[update_mask]
            self.momentum_mean_square[update_mask] = new_second_momentum[update_mask]
            mmt_mean_hat = self.momentum_mean/(1-momentum**self.global_step[:,None])
            mmt_mean_sq_hat = self.momentum_mean_square/(1-second_momentum**self.global_step[:,None])
            g_mean = mmt_mean_hat/(torch.sqrt(mmt_mean_sq_hat)+eps)
            self.global_step[update_mask] += 1
        g_mean[not_update_mask] = 0
        self.means -= lr*g_mean
        new_certainty = momentum*self.certainty + (1-momentum)*self.cache["covs"]/(self.cache["posterior"]+self.epsilon).unsqueeze(dim = 1)/(1-momentum**self.global_step[:,None])
        self.certainty[update_mask] = new_certainty[update_mask]
        if not self.fix_cov:
            self.cov = self.certainty
        self._initialize_sufficient_statistics()
        return g_mean

class RHMM(torch.nn.Module):
    def __init__(self, 
                 emission_module:torch.nn.Module, 
                 transition_module:torch.nn.Module = None,
                 device:str = None,
                 normalize_transition:bool = False,
                 transition_operation:str = "sparse",
                 index_mapping:List[np.array] = None):
        """
        The Restricted hidden Markov model (RHMM), like the Non-homogenous HMM,
        that is the transition matrix of the model is varying among time, but 
        different from NHMM is that the transition matrix is restricted to 
        small amount of transitions for specific time t.

        Parameters
        ----------
        emission_module: torch.nn.Module
            The emission model to use, need to be a Module itself.
        transition_module: torch.nn.Module, optional
            The transition model, can be None as the transition matrix can be
            provided externally.
        device : str
            The device to run the model, can be cpu or cuda:0
        normalize_transition : bool, optional
            If we want to normalize the transition matrix, default is False.
        transition_operation: str, optional default is sparse
            The way of transition multiplication, default is sparse using
            sparse matrix operation, can be dense or compact.
        index_mapping: List[np.array], optional
            A list of index mapping matrix [from_map_matrix,to_map_matrix] can 
            be given if the transition matrix is given in a compact form.
        """
        super(RHMM, self).__init__()
        self.add_module("emission",emission_module)
        self.n_states = self.emission.N
        self.device = self._get_device(device)
        self.emission.device = self.device
        self.emission.to(device)
        self.epsilon = 1e-6
        self.log_epsilon = torch.log(torch.tensor(self.epsilon)) 
        self.normalization = normalize_transition
        possible_operations = ["sparse","dense","compact"]
        max_funcs = {"sparse":self.transition_max_sparse,
                               "dense":self.transition_max,
                               "compact":self.transition_max_compact}
        forward_funcs = {"sparse":self.transition_prob_sparse,
                                   "dense":self.transition_prob,
                                   "compact":self.transition_prob_compact}
        backward_funcs = {"sparse":self.transition_backward_sparse,
                          "dense":self.transition_backward,
                          "compact":self.transition_backward_compact}
        if transition_operation not in possible_operations:
            raise ValueError("Transition operation type can only be "+ ','.join(possible_operations)+" but %s is given"%(transition_operation))
        self.transition_operation = transition_operation
        if self.transition_operation == "compact":
            if index_mapping is None:
                raise ValueError("Index mapping is required for compact transition operation.")
            elif len(index_mapping) != 2:
                raise ValueError("Require two index mapping matrix, map_from, map_to.")
            else:
                source_mapping, target_mapping = index_mapping
                self.source_mapping = torch.from_numpy(source_mapping).long().to(device)
                self.target_mapping = torch.from_numpy(target_mapping).long().to(device)
        if transition_module:
            self.add_module("transition",transition_module)
        self.to(self.device)
        
    def forward(self, observation:torch.Tensor, 
                duration:torch.Tensor, 
                transition:List,
                start_prob:torch.Tensor = None) -> np.array:
        """
        The forward algorithm for gradient descent training.
        
        Parameters
        ----------
        observation : torch.Tensor with shape [B,T,K]
            Tensor gives a batch of observatipns where B is the batch size, 
            T is the time and K is the feature.
        duration: torch.Tensor with shape [B]
            Gives the duration of each observation.
        transition : A length L list and each element is torch.sparse_coo_tensor 
            with shape [B,N,N] gives the transition matrix at certain batch id, 
            where N is the number of states, first N index refers to the source 
            and second index refers to the target.
        start_prob: torch.Tensor with shape [B,N], optional.
            Give the start probability for the signal in batches, if None a 
            uniform distribution will be used.
        """
        log_alpha = self._forward(observation, transition,start_prob)
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = torch.gather(log_sums, 1, duration.view(-1,1) - 1)
        return log_probs
    
    def _forward(self, observation:torch.Tensor, 
                transition:np.array,
                start_prob:torch.Tensor) -> np.array:
        """
        Caculate the forward messages given a observation and transtion matrix,
        the log probability is returned.

        Parameters
        ----------
        observation : torch.Tensor with shape [B,T,K]
            Tensor gives a batch of observatipns where B is the batch size, 
            T is the time and K is the feature.
        transition : A length L list and each element is torch.sparse_coo_tensor 
            with shape [B,N,N] gives the transition matrix at certain batch id, 
            where N is the number of states, first N index refers to the source 
            and second index refers to the target.
        start_prob: torch.Tensor with shape [B,N]
            Give the start probability for the signal in batches.
            
        Returns
        -------
        log_alpha: torch.Tensor with shape [B,T,N]
            The log forward probability

        """
        batch_size,T,K = observation.shape     
        log_alpha = torch.zeros((batch_size,T,self.n_states),device = self.device)
        if start_prob is None:
            if transition[0].layout == torch.strided:
                start_prob = torch.sum(transition[0],dim = 2) + self.epsilon
            elif transition[0].layout == torch.sparse_coo:
                start_prob = torch.sparse.sum(transition[0],dim = 2).to_dense() + self.epsilon
            else:
                raise TypeError("Transition tensor need to be either dense or coo sparse tensor.")
        log_alpha[:,0,:] = self.emission(observation[:,0,:]) + start_prob.log()
        emission = self.emission(observation)
        for t in np.arange(1,T):
            if self.transition_operation == "sparse":
                assert not self.normalization, "Can't enable sparse operation will normalization is on."
                #TODO: implement nomalization for sparse operation.
                log_alpha[:,t,:] = emission[:,t,:] + self.transition_prob_sparse(transition[t],log_alpha[:,t-1,:])
            elif self.transition_operation == "dense":
                curr_transition = self.to_dense(transition[t])
                log_alpha[:,t,:] = emission[:,t,:] + self.transition_prob(curr_transition,log_alpha[:,t-1,:],normalization=self.normalization)
                del curr_transition
            elif self.transition_operation == "compact":
                n_batch,n_states,b2 = transition[t].shape
                curr_transition = transition[t][:,:,int(b2/2):]
                log_alpha[:,t,:] = emission[:,t,:] + self.transition_prob_compact(curr_transition,log_alpha[:,t-1,:],normalization=self.normalization)
                del curr_transition
            torch.cuda.empty_cache()
        return log_alpha
    
    def _backward(self, 
                  observation:torch.Tensor, 
                  transition:np.array,
                  duration:torch.Tensor,) -> np.array:
        """
        Calculate the backward messages given a observation and transition matrix
        Same parameters as the _forward method.
        
        Parameters
        ----------
        observation : torch.Tensor with shape [B,T,K]
            Tensor gives a batch of observatipns where B is the batch size, 
            T is the time and K is the feature.
        transition : A length L list and each element is torch.sparse_coo_tensor 
            with shape [B,N,N] gives the transition matrix at certain batch id, 
            where N is the number of states, first N index refers to the source 
            and second index refers to the target.
        duration: torch.Tensor with shape [B]
            Gives the duration of each observation.
        
        Returns
        -------
        log_beta: torch.Tensor with shape [B,T,N]
            The log backward probability.
        """
        batch_size,T,K = observation.shape
        log_beta = torch.zeros((batch_size,T,self.n_states),device = self.device)
        log_beta[:,duration-1,:] = 1./self.n_states
        for t in np.arange(T-2,-1,-1):
            batch_mask = t<(duration-1)
            if self.transition_operation == "sparse":
                if self.normalization:
                    raise ValueError("Can't enable sparse operation will normalization is on.")
                update = self.transition_backward_sparse(transition[t],
                                                         self.emission(observation[:,t+1,:]),
                                                         log_beta[:,t+1,:])
                log_beta[batch_mask,t,:] = update[batch_mask,:]
            elif self.transition_operation == "dense":
                curr_transition = self.to_dense(transition[t])[batch_mask,:,:]
                log_beta[batch_mask,t,:] = self.transition_backward(curr_transition,
                                                                    self.emission(observation[batch_mask,t+1,:]),
                                                                    log_beta[batch_mask,t+1,:],
                                                                    normalization = self.normalization)
                del curr_transition
            elif self.transition_operation == "compact":
                n_batch,n_states,b2 = transition[t].shape
                curr_transition = transition[t][batch_mask,:,:int(b2/2)]
                log_beta[batch_mask,t,:] = self.transition_backward_compact(curr_transition,
                                                                            self.emission(observation[batch_mask,t+1,:]),
                                                                            log_beta[batch_mask,t+1,:],
                                                                            normalization = self.normalization)
                del curr_transition
            torch.cuda.empty_cache()
        return log_beta
    
    def expectation(self, observation:torch.Tensor, 
                    duration:torch.Tensor,
                    transition:np.array,
                    start_prob:torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the posterior probability P(z_i|X) using forward-backward 
        algorithm.
        
        Returns
        -------
        gamma: torch.Tensor with shape [B,T,N]
            The normalized posterior probability P(z_i|X)
        """
        batch_size,T,K = observation.shape
        log_alpha = self._forward(observation,transition,start_prob)
        log_beta = self._backward(observation,transition,duration)
        log_gamma = log_alpha + log_beta
        # print("log_alpha:",log_alpha)
        # print("log_beta:",log_beta)
        gamma = torch.softmax(log_gamma,dim = 2)
        if torch.sum(gamma.sum(dim = 2)==0):
            raise ValueError("gamma has pure 0 entry.")
        mask = torch.arange(start = 1, end = T+1,device = self.device).repeat(batch_size,1)>(duration.unsqueeze(dim = 1))
        gamma[mask] = 0
        # print("gamma: ",gamma)
        return gamma
    
    def sampling(self,
                 path:torch.Tensor,
                 certainty:float = None):
        """
        Sampling a observation given the path

        Parameters
        ----------
        path : torch.Tensor, shape [B,T]
            A batch of the kmer sequence to sampling from.
        certainty: float, optional, default is None.
            A uniform certainty to use to generate the sample, if None, the
            uncertainty parameter of the emission model will be used.
        Returns
        -------
        A sampled signal with shape [B,T,K].

        """
        B,T = path.shape
        std = self.emission.uncertainty[path] if certainty is None else certainty
        return self.emission.means[path]+std*torch.rand((B,T,self.emssion.K))

    def viterbi_decode(self, observation:torch.Tensor, 
                       duration:torch.Tensor, 
                       transition:List,
                       start_prob:torch.Tensor = None) -> np.array:
        """
        Calculate the viterbi path.
        
        Parameters
        ----------
        observation : torch.Tensor with shape [B,T,K]
            Tensor gives a batch of observatipns where B is the batch size, 
            T is the time and K is the feature.
        duration: torch.Tensor with shape [B]
            Gives the duration of each observation.
        transition : A length L list and each element is torch.sparse_coo_tensor 
            with shape [B,N,N] gives the transition matrix at certain batch id, 
            where N is the number of states, first N index refers to the source 
            and second index refers to the target.
        start_prob: torch.Tensor with shape [B,N], optional.
            Give the start probability for the signal in batches, if None a 
            uniform distribution will be used.
        """
        batch_size,T,K = observation.shape     
        path = torch.zeros((batch_size,T,self.n_states),dtype = torch.long, device = self.device)
        log_delta = torch.zeros((batch_size,T,self.n_states),device = self.device)
        viterbi_path = torch.zeros((batch_size,T),dtype = torch.long,device = self.device)
        if start_prob is None:
            if transition[0].layout == torch.strided:
                start_prob = torch.sum(transition[0],dim = 2) + self.epsilon
            else:
                start_prob = torch.sparse.sum(transition[0],dim = 2).to_dense() + self.epsilon
        log_delta[:,0,:] = self.emission(observation[:,0,:]) + start_prob.log()
        for t in np.arange(1,T):
            if self.transition_operation == "sparse":
                max_prob,max_idx = self.transition_max_sparse(transition[t], log_delta[:,t-1,:])
            elif self.transition_operation == "dense":
                curr_transition = self.to_dense(transition[t])
                max_prob, max_idx = self.transition_max(curr_transition, log_delta[:,t-1,:],normalization=self.normalization)
                del curr_transition
            elif self.transition_operation == "compact":
                n_batch,n_states,b2 = transition[t].shape
                curr_transition = transition[t][:,:,int(b2/2):]
                max_prob, max_idx = self.transition_max_compact(curr_transition, log_delta[:,t-1,:],normalization=self.normalization)
                del curr_transition
            log_delta[:,t,:] = self.emission(observation[:,t,:]) + max_prob
            path[:,t,:] = max_idx
            torch.cuda.empty_cache()
        viterbi_path.scatter_(1,(duration-1).unsqueeze(1),torch.argmax(log_delta[np.arange(batch_size),duration-1,:],dim = 1,keepdim = True))
        for t in np.arange(T-2,-1,-1):
            mask = duration > (t+1)
            viterbi_path[mask,t] = torch.gather(path[:,t+1,:],dim = 1,index = viterbi_path[:,t+1][:,None])[mask[:,None]]
        return viterbi_path,log_delta
    
    def transition_backward(self,
                            transition:torch.Tensor,
                            log_emission:torch.Tensor,
                            log_beta:torch.Tensor,
                            normalization:bool = True):
        """
        Calculate the step backward log probability.
        log beta_t^k = log sum_i exp{log T_t^{k,i} + log e^i(x_{t+1})+log beta_{t+1}^i}

        Parameters
        ----------
        transition : torch.Tensor shape [B,N,N]
            The transition matrix at current time point.
        log_emission : torch.Tensor shape [B,N]
            The log emission probability at time point t+1.
        log_beta : torch.Tensor shape [B,N]
            The t+1 log beta posterior probability.
        normalization : bool, opotional
            If we want to normalize the transition matrix.

        Returns
        -------
        The log beta posterior probabtiliy at current time point t.

        """
        if normalization:
            log_transition = normalize(transition,p=1,dim=2).log()
        else:
            log_transition = transition.log()
        elementwise_sum = log_transition + log_emission.unsqueeze(dim = 1) + log_beta.unsqueeze(dim = 1)
        return torch.logsumexp(elementwise_sum, dim=2)
    
    def transition_backward_sparse(self,
                                   transition:torch.sparse_coo_tensor,
                                   log_emission:torch.Tensor,
                                   log_beta:torch.Tensor):
        """
        Calculate the step backward log probability.
        log beta_t^k = log sum_i exp{log T_t^{k,i} + log e^i(x_{t+1})+log beta_{t+1}^i}

        Parameters
        ----------
        transition : torch.sparse_coo_tensor shape [B,N,N]
            The transition matrix at current time point.
        log_emission : torch.Tensor shape [B,N]
            The log emission probability at time point t+1.
        log_beta : torch.Tensor shape [B,N]
            The t+1 log beta posterior probability.
        normalization : bool, opotional
            If we want to normalize the transition matrix.

        Returns
        -------
        The log beta posterior probabtiliy at current time point t.

        """
        B = log_emission.unsqueeze(dim = 1) + log_beta.unsqueeze(dim = 1)
        result = log_domain_sparse_matmul(transition, B,dim = 2).to_dense()
        result[result == 0] = self.log_epsilon.to(self.device) + B.squeeze(dim = 1)[result == 0]
        return result
    
    def transition_backward_compact(self,
                                    transition:torch.Tensor,
                                    log_emission:torch.Tensor,
                                    log_beta:torch.Tensor,
                                    normalization:bool = False):
        """
        Calculate the step backward log probability.
        log beta_t^k = log sum_i exp{log T_t^{k,i} + log e^i(x_{t+1})+log beta_{t+1}^i}

        Parameters
        ----------
        transition : torch.sparse_coo_tensor shape [B,N,b+1]
            The compact transition matrix at current time point.
        log_emission : torch.Tensor shape [B,N]
            The log emission probability at time point t+1.
        log_beta : torch.Tensor shape [B,N]
            The t+1 log beta posterior probability.
        normalization : bool, opotional
            If we want to normalize the transition matrix.

        Returns
        -------
        The log beta posterior probabtiliy at current time point t.

        """
        if normalization:
            log_transition = normalize(transition,p=1,dim=2).log()
        else:
            log_transition = transition.log()
        B = log_emission + log_beta
        multiplication = B[:,self.target_mapping] #resulting shape [B,N,b+1]
        return torch.logsumexp(log_transition + multiplication,dim = -1)
    
    def transition_prob(self, 
                        transition_matrix:torch.Tensor, 
                        log_alpha:torch.Tensor,
                        normalization:bool = True):
        """
        Calculate the transition probability given the current transition
        matrix and current forward probability.

        Parameters
        ----------
        transition_matrix : torch.Tensor wish shape [B,N,N]
            Current transition matrix, B is the batch size and N is the number
            of states.
        log_alpha : torch.Tensor with shape [B,N]
            Current time alpha matrix. The default is None.
        normalization : bool, optional
            If we want to normalize the transition matrix.
        Returns
        -------
        The log transition probability with shape [B,N].

        """
        if normalization:
            log_transition = normalize(transition_matrix,p=1,dim=2).log()
        else:
            log_transition = transition_matrix.log()
        return log_domain_matmul(log_alpha, log_transition)
    
    def transition_prob_sparse(self, 
                               transition_matrix:torch.sparse_coo_tensor, 
                               log_alpha:torch.Tensor):
        """
        Calculate the transition probability given the current transition
        matrix and current forward probability.

        Parameters
        ----------
        transition_matrix : torch.sparse_coo_tensor wish shape [B,N,N]
            Current transition matrix, B is the batch size and N is the number
            of states.
        log_alpha : torch.Tensor with shape [B,N]
            Current time alpha matrix. The default is None.
        Returns
        -------
        The log transition probability with shape [B,N].

        """
        result = log_domain_sparse_matmul(transition_matrix, log_alpha.unsqueeze(dim = 2), dim = 1).to_dense()
        result[result == 0] = self.log_epsilon.to(self.device) + log_alpha[result == 0]
        return result
    
    def transition_prob_compact(self,
                                transition_matrix:torch.Tensor, 
                                log_alpha:torch.Tensor,
                                normalization:bool = False):
        """
        Calculate the transition probability given the compact transition
        matrix and current forward probability.

        Parameters
        ----------
        transition_matrix : torch.Tensor wish shape [B,N,b+1]
            Current transition matrix, B is the batch size and N is the number
            of states, b is the number of base.
        log_alpha : torch.Tensor with shape [B,N]
            Current time alpha matrix. The default is None.
        normalization : bool, optional
            If we want to normalize the transition matrix.
        Returns
        -------
        The log transition probability with shape [B,N].

        """
        if normalization:
            log_transition = normalize(transition_matrix,p=1,dim=2).log()
        else:
            log_transition = transition_matrix.log()
        indexing = self.source_mapping
        multiplication = log_alpha[:,indexing]
        return torch.logsumexp(log_transition + multiplication,dim = -1)
    
    def transition_max(self,
                        transition_matrix:torch.Tensor, 
                        log_delta:torch.Tensor,
                        normalization:bool = True):
        """
        Calculate the transition probability of the most probable path given 
        the current transition matrix and current forward probability.

        Parameters
        ----------
        transition_matrix : torch.Tensor wish shape [B,N,N]
            Current transition matrix, B is the batch size and N is the number
            of states.
        log_delta : torch.Tensor with shape [B,N]
            Current time alpha matrix.
        normalization : bool, optional
            If we want to normalize the transition matrix.
        Returns
        -------
        (log_prob, max_idx)
        The log transition probability of most probable path and the argmax 
        index.

        """
        if normalization:
            log_transition = normalize(transition_matrix,p=1,dim=2).log()
        else:
            log_transition = transition_matrix.log()
        return torch.max(log_delta.unsqueeze(dim = 2) + log_transition, dim = 1)
    
    def transition_max_compact(self,
                               transition_matrix:torch.Tensor, 
                               log_delta:torch.Tensor,
                               normalization:bool = True):
        """
        Calculate the transition probability of the most probable path given 
        the current transition matrix and current forward probability.

        Parameters
        ----------
        transition_matrix : torch.Tensor wish shape [B,N,N]
            Current transition matrix, B is the batch size and N is the number
            of states.
        log_delta : torch.Tensor with shape [B,N]
            Current time alpha matrix.
        normalization : bool, optional
            If we want to normalize the transition matrix.
        Returns
        -------
        (log_prob, max_idx)
        The log transition probability of most probable path and the argmax 
        index.

        """
        if normalization:
            log_transition = normalize(transition_matrix,p=1,dim=2).log()
        else:
            log_transition = transition_matrix.log()
        indexing = self.source_mapping
        multiplication = log_delta[:,indexing]
        max_prob,max_idx = torch.max(log_transition + multiplication,dim = -1)
        source_mapping_ex = self.source_mapping.expand(max_idx.shape[0],*self.source_mapping.shape)
        max_idx = source_mapping_ex.gather(-1,max_idx.unsqueeze(-1)).squeeze(-1)
        return max_prob,max_idx
    
    def transition_max_sparse(self,
                        transition_matrix:torch.Tensor, 
                        log_delta:torch.Tensor):
        """
        Calculate the transition probability of the most probable path given 
        the current transition matrix and current forward probability.

        Parameters
        ----------
        transition_matrix : torch.Tensor wish shape [B,N,N]
            Current transition matrix, B is the batch size and N is the number
            of states.
        log_delta : torch.Tensor with shape [B,N]
            Current time alpha matrix.
        Returns
        -------
        (log_prob, max_idx)
        The log transition probability of most probable path and the argmax 
        index.

        """
        B,N = log_delta.shape
        result,max_idx = log_domain_sparse_max(transition_matrix, log_delta.unsqueeze(dim = 2), dim = 1)
        max_idx[result == 0] = torch.arange(N,device = self.device).repeat(B,1)[result == 0]
        result[result == 0] = self.log_epsilon.to(self.device) + log_delta[result == 0]
        return result,max_idx
        
    def _to_device(self,data,device):
        if isinstance(data, (list,tuple)):
            return [self._to_device(x,device) for x in data]
        if isinstance(data, (dict)):
            temp_dict = {}
            for key in data.keys():
                temp_dict[key] = self._to_device(data[key],device)
            return temp_dict
        return data.to(device, non_blocking=True)
    
    def _get_device(self,device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    
    
    def maximization(self, observation:torch.Tensor, 
                     marginal_probability:torch.Tensor,
                     update = False,
                     lr = 0.1,
                     weight_decay = 0.0,
                     momentum = 0.9):
        self.emission._accumulate_sufficient_statistics(observation,
                                                        marginal_probability)
        if update:
            return self.emission.update_parameters(lr = lr,
                                                   weight_decay = weight_decay,
                                                   momentum = momentum)
    
    def to_dense(self, sparse_matrix_array:Union[np.array,torch.sparse_coo_tensor]):
        """
        Make an array of sparse matrix into a dense 3D tensor.
    
        Parameters
        ----------
        sparse_matrix_array : Union[np.array,torch.sparse_coo_tensor]
            A 1D np array, each element is a torch sparse tensor or a sparse torch
            coo tensor with shape [B,N,N]
        epsilon : float
            A small number to add to diagonal to make numerical stability.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        n_states = sparse_matrix_array[0].size()[0]
        if type(sparse_matrix_array) == np.ndarray:
            dense_array = torch.stack([x.to_dense() for x in sparse_matrix_array])
            if dense_array.device != self.device:
                dense_array = dense_array.to(self.device)
            return dense_array + self.epsilon * torch.eye(n_states,device = self.device)
        else:
            return sparse_matrix_array.to_dense() + (self.epsilon * torch.eye(n_states,device = self.device))

def log_domain_matmul(log_A, log_B):
    """
    log_A : m x n
    log_B : m x n x p or n x p
    output : m x p matrix
    
    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{i,k,j}

	A log domain matrix multiplication
	computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{i,k,j}
	"""
    dim_B = len(log_B.shape)
    log_A = log_A.unsqueeze(dim = 2)
    if dim_B == 2:
        log_B = log_B.unsqueeze(dim = 0)
    elementwise_sum = log_A + log_B
    out = torch.logsumexp(elementwise_sum, dim=1)
    return out

def log_domain_matmul_mapping(log_A, log_B, index_map):
    """
    log_A : m x n
    log_B : m x p x b or p x b
    index_map : p x b
    output : m x p matrix

	The following log domain matrix multiplication with index mapping is calculated
	computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{i,k,j}
	"""
    dim_B = len(log_B.shape)
    log_A = log_A.unsqueeze(dim = 2)
    if dim_B == 2:
        log_B = log_B.unsqueeze(dim = 0)
    elementwise_sum = log_A + log_B
    out = torch.logsumexp(elementwise_sum, dim=1)
    return out

def density(sparse_tensor:torch.sparse_coo_tensor):
    return len(sparse_tensor.indices())/torch.numel(sparse_tensor)

def log_domain_sparse_matmul(A:torch.sparse_coo_tensor, log_B:torch.Tensor, dim:int = -1):
    """
    Do a sparse-dense tensor multiplication and reduced on the given dimension.

    Parameters
    ----------
    A : torch.sparse_coo_tensor
        A sparse tensor with shape mxnxp.
    log_B : torch.Tensor
        A dense tensor in the log domain with same shape, broadcast is supported on dense tensor.
    dim : int, optional
        The dimension to perform reduction on A. The default is -1.

    Returns
    -------
    A sparse tensor.

    """
    idxs,update = log_domain_sparse_product(A,log_B)
    n_dims_A = idxs.shape[0]
    shape_A = torch.tensor(A.shape)
    remain_dims = np.arange(n_dims_A)
    remain_dims = np.delete(remain_dims,dim)
    remain_idxs = idxs[remain_dims,:].T.tolist()
    update = update.tolist()
    key_func = lambda x: x[1]
    update = sorted(zip(update,remain_idxs),key = key_func)
    nested = [ (k,list(g)) for k,g in itertools.groupby(update,key = key_func)]
    nested_vals = [[y[0] for y in x[1]] for x in nested]
    nested_idxs = [x[0] for x in nested]
    max_cols = max([len(x) for x in nested_vals])
    padded = torch.tensor([x + [-np.inf]*(max_cols - len(x)) for x in nested_vals],device = A.device)
    return torch.sparse_coo_tensor(indices = list(zip(*nested_idxs)),
                                   values = torch.logsumexp(padded,dim = 1),
                                   size = shape_A[remain_dims].tolist(),
                                   device = A.device)

def log_domain_sparse_max(A:torch.sparse_coo_tensor, log_B:torch.Tensor, dim:int = -1):
    """
    Do a sparse-dense tensor multiplication and reduced on the given dimension.

    Parameters
    ----------
    A : torch.sparse_coo_tensor
        A sparse tensor with shape mxnxp.
    log_B : torch.Tensor
        A dense tensor in the log domain with same shape, broadcast is supported on dense tensor.
    dim : int, optional
        The dimension to perform reduction on A. The default is -1.

    Returns
    -------
    A sparse tensor.

    """
    idxs,update = log_domain_sparse_product(A,log_B)
    n_dims_A = idxs.shape[0]
    shape_A = torch.tensor(A.shape)
    remain_dims =np.delete(np.arange(n_dims_A),dim)
    remain_idxs = list(zip(*[idxs[x].tolist() for x in remain_dims]))
    reduced_idxs = idxs[dim].tolist()
    update = update.tolist()
    key_func = lambda x: x[2]
    update = sorted(zip(update,reduced_idxs,remain_idxs),key = key_func)
    nested = [ (k,list(g)) for k,g in itertools.groupby(update,key = key_func)]
    nested_vals = [[y[0] for y in x[1]] for x in nested]
    nested_idxs = list(zip(*[x[0] for x in nested]))
    nested_reduced_idxs = [[y[1] for y in x[1]] for x in nested]
    max_cols = max([len(x) for x in nested_vals])
    padded = torch.tensor([x + [-np.inf]*(max_cols - len(x)) for x in nested_vals],device = A.device)
    padded_reduced_idxs = torch.tensor([x + [-1]*(max_cols - len(x)) for x in nested_reduced_idxs],device = A.device,dtype = torch.long)
    result,argmax = torch.max(padded,dim = 1)
    argmax = torch.gather(padded_reduced_idxs,dim = 1,index = argmax[:,None]).squeeze(dim = 1)
    max_idx = torch.zeros(shape_A[remain_dims].tolist(),device = A.device,dtype = torch.long)
    max_idx[tuple(nested_idxs)] = argmax
    return torch.sparse_coo_tensor(indices = nested_idxs,
                                   values = result,
                                   size = shape_A[remain_dims].tolist(),
                                   device = A.device).to_dense(),max_idx


def log_domain_sparse_product(A:torch.sparse_coo_tensor, log_B:torch.Tensor):
    """
    Do a sparse-dense tensor production.

    Parameters
    ----------
    A : torch.sparse_coo_tensor
        A sparse tensor with shape mxnxp.
    log_B : torch.Tensor
        A dense tensor in the log domain with same shape, broadcast is supported on dense tensor.

    Returns
    -------
    idxs,vals

    """
    A = A.coalesce()
    idxs = A.indices()
    log_vals = A.values().log()
    shape_A = torch.tensor(A.shape)
    shape_B = torch.tensor(log_B.shape)
    n_dims_A = idxs.shape[0]
    n_dims_B = len(shape_B)
    assert n_dims_A == n_dims_B, "Tensor has different number of dimensions."
    assert torch.all(torch.logical_or(shape_A == shape_B,shape_B == 1)), "Shape mismatch, got {} and {}. Broadcast only supported on dense tensor.".format(shape_A,shape_B)
    idxs_B = idxs.clone()
    idxs_B[torch.where(shape_B==1)] = 0
    update = log_B[tuple(idxs_B)] + log_vals
    return idxs,update
    
    
if __name__ == "__main__":
    # Module test
    from xron.nrhmm.hmm_input import Kmer_Dataset
    from torch.utils.data.dataloader import DataLoader
    from xron.xron_train_base import DeviceDataLoader
    from matplotlib import pyplot as plt
    import time
    ## Construct the simulation data
    np.random.seed(2022)
    device = "cuda"
    n_states = 10
    background = 0.1
    T_max = 20000
    n_samples = 1500
    batch_size = 50
    single_length = 500
    segment_length = 20
    epoches_n = 100
    means = np.random.rand(n_states) * 5
    init_means = np.random.rand(n_states)
    covs = np.random.rand(n_states)
    init_covs = np.asarray([2]*n_states)
    z_true = np.asarray([[x]*segment_length for x in np.random.randint(0,n_states,T_max//segment_length)]).flatten()
    x = means[z_true] + covs[z_true]*(np.random.rand(T_max)-0.5)*2
    
    transitions = np.empty(T_max,dtype = object)
    for t in np.arange(T_max-1):
        source = [z_true[max(t-2*segment_length,0)],z_true[max(t-2*segment_length,0)],z_true[max(t-segment_length,0)],z_true[max(t-segment_length,0)],z_true[t],z_true[t],z_true[min(t+segment_length,T_max-2)],z_true[min(t+segment_length,T_max-2)],z_true[min(t+2*segment_length,T_max-2)]]
        target = [z_true[max(t-2*segment_length,0)],z_true[max(t-segment_length,0)],z_true[max(t-segment_length,0)],z_true[t],z_true[t],z_true[min(t+segment_length,T_max-2)],z_true[min(t+segment_length,T_max-2)],z_true[min(t+2*segment_length,T_max-2)],z_true[min(t+2*segment_length,T_max-2)]]
        source,target = list(zip(*set(list(zip(source,target)))))
        curr_transition = torch.sparse_coo_tensor([source,target], [1.0]*len(source),(n_states,n_states))
        transitions[t] = curr_transition
    transitions[T_max-1] = torch.sparse_coo_tensor([[z_true[-1]],[z_true[-1]]],[1],(n_states,n_states))
    def sampling(x, n_samples,length = single_length):
        T_max = len(x)
        start = np.random.randint(low = 0,high = T_max-length, size = n_samples)
        idxs = np.asarray([np.arange(x,x+length) for x in start])
        return idxs, x[idxs]
    idxs,xs = sampling(x,n_samples)
    xs = xs.astype(np.float32)
    Transitions = transitions[idxs].T
    list_transitions = [list(x) for x in list(Transitions.T)]
    labels = z_true[idxs]
    starts = np.zeros((n_samples,n_states)) + background
    starts[range(n_samples),z_true[idxs[:,0]]] = 1.
    Ls = np.asarray([single_length]*n_samples)

    # Construct the model and training using the direct gradient descent method.
    # emission = GaussianEmissions(init_means[:,None], init_covs[:,None])
    # hmm = RHMM(emission,normalize_transition=False,sparse_operation=False)
    # optimizer = torch.optim.Adam(hmm.parameters(),lr = 1e-1)
    # perm = np.arange(n_samples)
    # update_every_n = 5
    # dataset = Kmer_Dataset(xs, Ls, list_transitions)
    # loader = DataLoader(dataset,batch_size = batch_size, shuffle = False)
    # loader = DeviceDataLoader(loader,device = device)
    # for epoch_i in np.arange(epoches_n):
    #     for i,batch in enumerate(loader):
    #         batch_x = batch['signal']
    #         transition = batch['labels']
    #         start = torch.zeros((batch_size,n_states)).to(device)
    #         batch_label = labels[(i)*batch_size%n_samples:(((i+1)*batch_size-1)%n_samples+1)]
    #         batch_duration = batch['duration']
    #         log_prob = hmm.forward(batch_x, batch_duration, transition)
    #         loss = -log_prob.mean()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         hmm.emission.clamp_cov()
    #         error = np.linalg.norm(emission.means.cpu().squeeze(dim = 1).detach().numpy() - means,ord = 1)/n_states
            
    #         path,logit = hmm.viterbi_decode(batch_x, batch_duration, transition)
    #         mask = np.arange(single_length)[None,:].repeat(batch_x.shape[0],axis = 0)<batch_duration[:,None].detach().cpu().numpy()
    #         accr = (batch_label[mask] == path.detach().cpu().numpy()[mask]).sum()/(sum(batch_duration.detach().cpu().numpy()))
    #         print("%d epoch-%d batch: loss %.2f, error %.2f, accuracy %.2f"%(epoch_i,i,loss.cpu().detach().numpy(),error,accr))

    # Train using Expectation-Maximization algorithm
    visual = False
    emission = GaussianEmissions(init_means[:,None], init_covs[:,None])
    hmm = RHMM(emission,normalize_transition=False)
    optimizer = torch.optim.Adam([hmm.emission.means],lr = 1)
    perm = np.arange(n_samples)
    update_every_n = 1
    dataset = Kmer_Dataset(xs, Ls, list_transitions,labels = labels)
    loader = DataLoader(dataset,batch_size = batch_size, shuffle = True)
    loader = DeviceDataLoader(loader,device = device)
    for epoch_i in np.arange(epoches_n):
        for i,batch in enumerate(loader):
            start_time = time.time()
            batch_x = batch['signal']
            transition = batch['labels']
            batch_label = batch['y']
            start = torch.zeros((batch_size,n_states)).to(device)
            batch_duration = batch['duration']
            with torch.no_grad():
                s1 = time.time()
                gamma = hmm.expectation(batch_x, batch_duration, transition)
                print("Expectation time %.2f"%(time.time() - s1))
                g = hmm.maximization(batch_x,gamma,update = (i%update_every_n==0),momentum = 0.9,lr = 1. ,weight_decay = 0.0)
            if i%update_every_n == 0:
                error = np.linalg.norm(emission.means.cpu().squeeze(dim = 1).detach().numpy() - means,ord = 1)/n_states
                log_prob = hmm.forward(batch_x, batch_duration, transition)
                loss = -torch.logsumexp(log_prob, dim = 1).mean()
                path,logit = hmm.viterbi_decode(batch_x, batch_duration, transition)
                mask = np.arange(single_length)[None,:].repeat(batch_size,axis = 0)<batch_duration[:,None].detach().cpu().numpy()
                accr = (batch_label[mask].detach().cpu().numpy() == path.detach().cpu().numpy()[mask]).sum()/(sum(batch_duration.detach().cpu().numpy()))
                print("Epoch %d - Batch %d: loss %.2f, error %.2f, accuracy %.2f, time per batch %.2f"%(epoch_i,i,loss,error,accr,time.time()-start_time))
                max_variance = np.max(hmm.emission.certainty.cpu().numpy())
                print("Rescale the covariance to %.2f"%(max_variance))
                hmm.emission.cov[:] = 2*max_variance
                # print("Epoch %d - Batch %d: error %.2f, accuracy %.2f"%(epoch_i,i,error,accr))
                if visual:
                    plt.plot(logit[0,np.arange(single_length),batch_label[0]].detach().cpu().numpy(),label = "Oirignal.")
                    plt.plot(logit[0,np.arange(single_length),path[0]].detach().cpu().numpy(),label = "Realigned.")
                    plt.ylabel("Log probability.")
                    plt.xlabel("Time")
                    plt.figure()
                    plt.plot(batch_x[0].cpu().numpy())
                    rc_sig = [hmm.emission.means[x].item() for x in path[0]]
                    plt.plot(rc_sig)
                    