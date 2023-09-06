#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 04:38:02 2022

@author: haotian teng
"""
import torch
import itertools
import numpy as np

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
    shape_A = torch.tensor(A.shape)
    remain_dims = np.delete(np.arange(idxs.shape[0]),dim)
    remain_idxs = idxs[remain_dims,:].T.tolist()
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

def log_domain_sparse_matmul_new(A:torch.sparse_coo_tensor, log_B:torch.Tensor, dim:int = -1):
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
    shape_A = torch.tensor(A.shape)
    remain_dims = np.delete(np.arange(idxs.shape[0]),dim)
    remain_idxs = idxs[remain_dims,:].T
    uniq,uniq_idxs = torch.unique(remain_idxs,dim = 0, return_inverse = True)
    nested_vals = [update[uniq_idxs == i].tolist() for i in np.arange(len(uniq))]
    # nested_vals = [1]*len(uniq)
    max_cols = max([len(x) for x in nested_vals])
    padded = torch.tensor([x + [-np.inf]*(max_cols - len(x)) for x in nested_vals],device = A.device)
    return torch.sparse_coo_tensor(indices = uniq.T,
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
    from time import time
    N = 3000
    K = 3125
    B = 100
    i = torch.tensor([np.random.randint(low = 0, high = B, size = N),
                      np.random.randint(low = 0, high = K, size = N),
                      np.random.randint(low = 0, high = K, size = N)])
    v = torch.tensor(np.random.rand(N), dtype=torch.float32)
    s = torch.sparse_coo_tensor(i, v, [B, K, K])
    d = torch.rand((B,K,1))
    start = time()
    result = log_domain_sparse_matmul(s,d,dim = 1)
    print(time()-start)
    start = time()
    result_new = log_domain_sparse_matmul_new(s,d,dim=1)
    print(start-time())
    
    idxs = s.coalesce().indices()
    remain_dims = np.delete(np.arange(idxs.shape[0]),1)
    remain_idxs = idxs[remain_dims].T