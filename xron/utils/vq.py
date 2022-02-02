"""
Created on Tue Dec 28 13:33:50 2021

@author: Haotian Teng
"""
import torch

class NearstEmbedding(torch.autograd.Function):
    """Get the nearest embedding of given input, used in VQ-VAE:
        https://arxiv.org/pdf/1711.00937.pdf
    """
    @staticmethod
    def forward(ctx,input,embd_weight):
        """
        Find the nearest neighbour of the embedding variable.

        Parameters
        ----------
        ctx : Object.
            A object to stash information for backward computation, save 
            arbitrary objects for use in backward computation by ctx.cave_for_backward method.
        input : torch.tensor
            The input tensor with size (*,embedding_dim)
        embd_weight : torch.tensor
            The embedding weight tensor with size (num_embeddings,embedding_dim)

        Returns
        -------
        The nearest embedding vector.

        """
        index = vq_idx(input,embd_weight)
        ctx.mark_non_differentiable(index)
        return torch.index_select(embd_weight,dim = 0,index = index).view_as(input)
    
    @staticmethod
    def backward(ctx,grad_post):
        """
        Propgate the gradient from posterior layer back to the previous layer,
        dL/dq = dL/de.
        """
        grad_input, grad_embd = None,None
        if ctx.needs_input_grad[0]:
            grad_input = grad_post.clone()
        return grad_input,grad_embd

class NearstEmbeddingIndex(torch.autograd.Function):
    """Get the indexs of the nearest embedding of given input, no gradient will
    be propagate back.
    """
    @staticmethod
    def forward(ctx,input,embd_weight):
        """
        Find the nearest neighbour of the embedding variable.

        Parameters
        ----------
        ctx : Object.
            A object to stash information for backward computation, save 
            arbitrary objects for use in backward computation by ctx.cave_for_backward method.
        input : torch.tensor
            The input tensor with size (*,embedding_dim)
        weight : torch.tensor
            The embedding weight tensor with size (num_embeddings,embedding_dim)

        Returns
        -------
        The nearest embedding vector.

        """
        with torch.no_grad():
            if input.shape[-1] != embd_weight.shape[-1]:
                raise ValueError("Input tensor shape %d is not consistent with the embedding %d."%(input.shape[-1],embd_weight.shape[-1]))
            distance = torch.cdist(input.flatten(end_dim = -2),embd_weight)
            index = torch.argmin(distance,dim = 1)
            ctx.mark_non_differentiable(index)
            return index
    
    @staticmethod
    def backward(ctx,grad_post):
        """
        No gradient since this is for training embedding.
        """
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`NearstEmbeddingIndex`. The function `NearstEmbeddingIndex` '
            'is not differentiable. Use `NearstEmbedding` '
            'if you want a straight-through estimator of the gradient.')

def vq_idx(x,embd_weight):    
    return NearstEmbeddingIndex().apply(x,embd_weight)

def vq(x, embd_weight):
    idx = NearstEmbeddingIndex().apply(x,embd_weight.detach())
    e_shadow = torch.index_select(embd_weight,dim = 0,index = idx).view_as(x)
    e = NearstEmbedding().apply(x, embd_weight.detach())
    return e,e_shadow

def loss(x,embd_weight):
    e,e_shadow = vq(x,embd_weight)
    mse_loss = torch.nn.MSELoss(reduction = "mean")
    sg_q = x.detach()
    sg_e = e.detach()
    rc_signal = e
    rc_loss = mse_loss(rc_signal,x)
    embedding_loss = mse_loss(sg_q,e_shadow)
    commitment_loss = mse_loss(sg_e,x)
    return rc_loss, embedding_loss, commitment_loss

if __name__ == "__main__":
    import torchviz
    N_EMBD = 10
    EMBD_DIM = 3
    torch.manual_seed(1992)
    embd = torch.nn.Embedding(N_EMBD,EMBD_DIM)
    embd.weight.data.uniform_(-1./N_EMBD, 1./N_EMBD)
    embd_weight = embd.weight
    x = torch.rand((2,2,EMBD_DIM),requires_grad=True, dtype = torch.float)
    
    idx = NearstEmbeddingIndex.apply(x,embd_weight)
    rc_loss, embedding_loss, commitment_loss= loss(x,embd_weight)
    grad_x,grad_embd = torch.autograd.grad((rc_loss, embedding_loss, commitment_loss), (x,embd_weight), create_graph=True)
    torchviz.make_dot((grad_x,grad_embd, x, embd_weight, rc_loss, embedding_loss, commitment_loss), 
                      params={"grad_x": grad_x, 
                              "grad_embd":grad_embd, 
                              "x": x, 
                              "embedding_weight":embd_weight, 
                              "rc_loss": rc_loss, 
                              "embedding_loss":embedding_loss, 
                              "commitment_loss":commitment_loss}).render("./all_loss", format="png")
    
    e = NearstEmbedding().apply(x,embd_weight)
    nn_grad ,_ = torch.autograd.grad((e.sum()), (x,embd_weight), create_graph=True,allow_unused=True )
    torchviz.make_dot((nn_grad, x, embd_weight,e), params={"grad_x": nn_grad, "x": x, "embedding_weight":embd_weight,"e":e}).render("./nearestembd", format="png")