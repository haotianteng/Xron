"""
Created on Thu Apr 28 18:03:07 2022

@author: Haotian Teng
"""
import torch
import itertools
import numpy as np
import timeit

def sparse_multiplication(transition,log_alpha):
    idxs = transition.indices()
    vals = transition.values().log()
    update_check = torch.zeros((batch_size,n_kmers),dtype = torch.bool)
    update = log_alpha[idxs[0],idxs[1]] +vals
    for i,u in enumerate(update):
        batch_i,kmer_i = idxs[0][i],idxs[2][i]
        if not update_check[batch_i,kmer_i]:
            curr_update = update[torch.logical_and(idxs[0]==batch_i,idxs[2] == kmer_i)]
            log_alpha[idxs[0][i],idxs[2][i]] = torch.logsumexp(curr_update,dim = 0)
            update_check[batch_i,kmer_i] = True
    log_alpha[torch.logical_not(update_check)] += np.log(1e-6)
    return log_alpha
        
def gpu_multiplication(transition,log_alpha):
    dense_t = transition.to_dense()
    return log_domain_matmul(log_alpha, dense_t.log())

def base_multiplication(transition_base,indexing,log_alpha):
    n_kmer,n_base = indexing.shape
    alpha_shape = log_alpha.shape
    multiplication = log_alpha[:,indexing]
    return torch.logsumexp(transition_base.log() + multiplication,dim = -1)

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
    A = A.coalesce()
    idxs = A.indices()
    log_vals = A.values().log()
    shape_A = torch.tensor(A.shape)
    shape_B = torch.tensor(log_B.shape)
    n_dims_A = idxs.shape[0]
    n_dims_B = len(shape_B)
    assert n_dims_A == n_dims_B, "Tensor has different number of dimensions."
    assert torch.all(shape_A >= shape_B), "Broadcast only supported on dense tensor."
    idxs_B = idxs.clone()
    remain_dims = np.arange(n_dims_A)
    remain_dims = np.delete(remain_dims,dim)
    remain_idxs = list(zip(*[idxs.cpu()[x].tolist() for x in remain_dims]))
    idxs_B[torch.where(shape_B==1)] = 0
    update = log_B[tuple(idxs_B)] + log_vals
    key_func = lambda x: x[1]
    update = update.tolist()
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
    sparsity = 0.0014
    n_kmers = 3125
    batch_size = 2
    n_base = 5
    n_elements = int(n_kmers*n_kmers*sparsity)
    start = np.random.randint(low = 0, high = n_kmers - 1, size = n_elements)
    end = np.random.randint(low = 0, high = n_kmers - 1, size = n_elements)
    batch_i = np.random.randint(low = 0, high = batch_size - 1, size = n_elements)
    transition = torch.sparse_coo_tensor(indices = (batch_i,start,end),values = [1.]*len(start),size = (batch_size,n_kmers,n_kmers)).coalesce()
    transition_cuda = transition.to("cuda")
    log_alpha = torch.rand((batch_size,n_kmers))
    log_alpha_cuda = log_alpha.to("cuda")
    result = log_domain_sparse_matmul(transition_cuda,log_alpha_cuda.unsqueeze(dim = 2),dim = 1)
    result_gpu = gpu_multiplication(transition_cuda,log_alpha_cuda)
    assert torch.all(torch.isclose(result.to_dense()[torch.logical_not(torch.isinf(result_gpu))], result_gpu[torch.logical_not(torch.isinf(result_gpu))]))
    # timeit.timeit(sparse_multiplication(transition, log_alpha),number = 10000)
    # timeit.timeit(gpu_multiplication(transition_cuda,log_alpha_cuda),number = 10000)
    transition_base = torch.rand((batch_size,n_kmers,n_base+1)).to("cuda")
    base_index = torch.randint(low = 0,high = n_kmers,size = (n_kmers,n_base + 1)).to("cuda")
    base_multiplication(transition_base,base_index,log_alpha_cuda)
    # timeit.timeit(base_multiplication(transition_base,base_index,log_alpha_cuda),number = 10000)