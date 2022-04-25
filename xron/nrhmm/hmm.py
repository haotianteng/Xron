"""
Created on Wed Mar 30 12:13:39 2022

@author: Haotian Teng
"""
import torch
from torch.nn.functional import normalize
import numpy as np
from typing import List,Union

class GaussianEmissions(torch.nn.Module):
    def __init__(self, means:np.array,cov:np.array, trainable:np.array = None):
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

        """
        super().__init__()
        self.N, self.K = means.shape
        self.means = torch.nn.Parameter(torch.from_numpy(means).float())
        self.trainable = torch.from_numpy(trainable).bool() if trainable else torch.Tensor([1]*self.N).bool()
        self.cov = torch.nn.Parameter(torch.from_numpy(cov).float(),requires_grad = True)
        self.cache = None #The cache when Baum-Welch is used.
        self.epsilon = 1e-6 # A small number to prevent numerical instability
        if trainable is not None:
            self.means.register_hook(lambda grad: grad*self.trainable[:,None])
            self.cov.register_hook(lambda grad: grad*self.trainable[:,None])
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Get the emission probability given a observation at time t.

        Parameters
        ----------
        x : torch.Tensor with shape [B,K]
            The observation value at certain time point, B is the batch size
            and K is the number of feature.

        Returns
        -------
        posterior: torch.Tensor with shape [B,N], return the log posterior 
        probability P(z|x)
        """
        return -0.5*((1/self.cov*(self.means-x[:,None,:])**2).sum(dim = 2)+torch.log(self.cov).sum(dim = 1)+self.K*np.log(2*np.pi))
    
    @torch.no_grad()
    def clamp_cov(self):
        """
        Clamp the covariance to positive number

        """
        self.cov[:] = torch.clamp(self.cov,min = self.epsilon)
    
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
    def update_parameters(self, moving_average:float = 0.9):
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
        a = moving_average
        assert 1>a>=0
        new_means = a*self.means + self.cache["means"]/(self.cache["posterior"]+self.epsilon).unsqueeze(dim = 1) *(1-a)
        new_cov = a*self.cov + self.cache["covs"]/(self.cache["posterior"]+self.epsilon).unsqueeze(dim = 1)*(1-a)
        self.trainable = self.trainable.to(self.device)
        update_mask = torch.logical_and(self.trainable,self.cache["posterior"]>0)
        self.means[update_mask,:] = new_means[update_mask,:]
        self.cov[update_mask,:] = new_cov[update_mask,:]
        # print("Updated covs:",self.cov)
        # print("Updated means:",self.means)
        self._initialize_sufficient_statistics()

class RHMM(torch.nn.Module):
    def __init__(self, 
                 emission_module:torch.nn.Module, 
                 transition_module:torch.nn.Module = None,
                 device:str = None,
                 normalize_transition:bool = True):
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
            If we want to normalize the transition matrix, default is True.

        """
        super(RHMM, self).__init__()
        self.add_module("emission",emission_module)
        self.n_states = self.emission.N
        self.device = self._get_device(device)
        self.emission.device = self.device
        self.epsilon = 1e-6
        self.normalization = normalize_transition
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
        curr_transition = self.to_dense(transition[0])
        if start_prob is None:
            start_prob = curr_transition.sum(dim = 2)
        log_alpha[:,0,:] = self.emission(observation[:,0,:]) + start_prob.log()
        del curr_transition
        for t in np.arange(1,T):
            curr_transition = self.to_dense(transition[t])
            log_alpha[:,t,:] = self.emission(observation[:,t,:]) + self.transition_prob(curr_transition,log_alpha[:,t-1,:],normalization=self.normalization)
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
            curr_transition = self.to_dense(transition[t])[batch_mask,:,:]
            log_beta[batch_mask,t,:] = self.transition_backward(curr_transition,
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
        None.

        """
        if normalization:
            log_transition = normalize(transition_matrix,p=1,dim=2).log()
        else:
            log_transition = transition_matrix.log()
        return log_domain_matmul(log_alpha, log_transition)
    
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
                     update = False):
        self.emission._accumulate_sufficient_statistics(observation,
                                                        marginal_probability)
        if update:
            self.emission.update_parameters()
        
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

if __name__ == "__main__":
    # Module test
    ## Construct the simulation data
    np.random.seed(2022)
    device = "cuda"
    n_states = 10
    background = 0.1
    T_max = 2000
    n_samples = 1500
    batch_size = 50
    single_length = 1000
    segment_length = 20
    epoches_n = 100
    means = np.random.rand(n_states) * 5
    init_means = np.random.rand(n_states)
    covs = np.random.rand(n_states)
    init_covs = np.asarray([1.]*n_states)
    z_true = np.asarray([[x]*segment_length for x in np.random.randint(0,n_states,T_max//segment_length)]).flatten()
    x = means[z_true] + covs[z_true]*(np.random.rand(T_max)-0.5)*2
    
    transitions = np.empty(T_max,dtype = object)
    for t in np.arange(T_max-1):
        source = [z_true[t]]
        target = [z_true[t+1]]
        curr_transition = torch.sparse_coo_tensor([source,target], [1.0],(n_states,n_states))
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
    starts = np.zeros((n_samples,n_states)) + background
    starts[range(n_samples),z_true[idxs[:,0]]] = 1.
    Ls = np.asarray([single_length]*batch_size)

    # Construct the model and training using the direct gradient descent method.
    # emission = GaussianEmissions(init_means[:,None], init_covs[:,None])
    # hmm = RHMM(emission,normalize_transition=False)
    # optimizer = torch.optim.Adam(hmm.parameters(),lr = 1e-1)
    # perm = np.arange(n_samples)
    # for i in np.arange(0,n_samples*epoches_n,batch_size):
    #     i = i%n_samples
    #     if i%n_samples == 0:
    #         np.random.shuffle(perm)
    #     choice = perm[i:i+batch_size]
    #     optimizer.zero_grad()
    #     batch_x = torch.from_numpy(xs[choice][:,:,None])
    #     start = torch.from_numpy(starts[choice,:])
    #     batch_duration = torch.from_numpy(Ls)
    #     # with torch.no_grad():
    #     log_prob = hmm.forward(batch_x, batch_duration, Transitions[:,choice])
    #     loss = -log_prob.mean()
    #     loss.backward()
    #     optimizer.step()
    #     hmm.emission.clamp_cov()
    #     error = np.linalg.norm(emission.means.cpu().squeeze(dim = 1).detach().numpy() - means,ord = 1)/n_states
    #     print("loss %.2f, error %.2f"%(loss.cpu().detach().numpy(),error))
    
    # Train using Expectation-Maximization algorithm
    emission = GaussianEmissions(init_means[:,None], init_covs[:,None])
    hmm = RHMM(emission,normalize_transition=False)
    optimizer = torch.optim.Adam(hmm.parameters(),lr = 1e-1)
    perm = np.arange(n_samples)
    update_every_n = 5*batch_size
    for i in np.arange(0,n_samples*epoches_n,batch_size):
        idx = i%n_samples
        if i%n_samples == 0:
            np.random.shuffle(perm)
        choice = perm[idx:idx+batch_size]
        batch_x = torch.from_numpy(xs[choice][:,:,None]).to(device)
        start = torch.from_numpy(starts[choice,:]).to(device)
        batch_duration = torch.from_numpy(Ls).to(device)
        with torch.no_grad():
            gamma = hmm.expectation(batch_x, batch_duration, Transitions[:,choice])
            hmm.maximization(batch_x,gamma,update = (i%update_every_n==0))
        if i%update_every_n == 0:
            error = np.linalg.norm(emission.means.cpu().squeeze(dim = 1).detach().numpy() - means,ord = 1)/n_states
            print("error %.2f"%(error))
    
    