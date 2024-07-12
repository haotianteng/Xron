import torch
import numpy as np
from xron.utils.seq_op import raw2seq
from torchaudio.functional import forced_align

def beam_search(decoder, logits, seq_len, return_paths = True,vocab = ['b','A','C','G','T','M']):
    results = decoder(logits, seq_len)
    tokens = [ [x.tokens for x in result] for result in results]
    seqs = [[''.join([vocab[x] for x in t]) for t in token] for token in tokens]
    if return_paths:
        moves, path_logit = force_align_batch(logits, tokens,device = "cpu")
        return tokens, moves, path_logit
    return tokens


def force_align_batch(log_probs, targets,device = 'cpu'):
    """A batch wrapper for force_align function
    force_align function can only take batch 1 as current version of torchaudio: 2.4.0
    log_probs: A tensor of shape [batch_size, max_time, num_classes]
    targets: A nested lists, where the first dimension is the batch size, and the second dimesnion 
        is n_best paths, and the third dimension is the tokens
    
    """
    moves, path_logits = [],[]
    log_probs = log_probs.to(device)
    for i in range(log_probs.shape[0]):
        target = targets[i]
        curr_m, curr_p = [],[]
        for t in target:
            t_len = torch.tensor([len(t)]).to(device)
            t = torch.tensor(t).to(device)
            align,pl = forced_align(log_probs = log_probs[i].unsqueeze(0),
                        targets = t.unsqueeze(0),
                        target_lengths=t_len)
            move = (align>0)
            curr_p.append(pl[move])
            curr_m.append(move.to(int).squeeze(0))
        moves.append(curr_m)
        path_logits.append(curr_p)    
    return moves, path_logits

def viterbi_decode(logits:torch.tensor):
    """
    Viterbi decdoing algorithm

    Parameters
    ----------
    logits : torch.tensor
        Shape L-N-C

    Returns
    -------
        sequence: A length N list contains final decoded sequence.
        moves: A length N list contains the moves array.

    """
    sequence = np.argmax(logits,axis = 2)
    sequence = sequence.T #L,N -> N,L
    sequence,moves = raw2seq(sequence)
    return sequence,list(moves.astype(int))

if __name__ == "__main__":
    from time import time
    from fast_ctc_decode import fast_beam_search
    from torchaudio.models.decoder import cuda_ctc_decoder,ctc_decoder
    T = 1000
    logits = torch.randn(5, T, 6)  # Example with batch size of 1, 100 time steps, and 6 classes
    logits[0,5:20,0] += 5

    # Convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Length of each sequence in the batch (assume all sequences are of max length for simplicity)
    seq_len = torch.tensor([T]*5, dtype=torch.int32)

    # Vocabulary (example)
    vocab = ['b','A','C','G','T','M']  # Example vocabulary

    # Initialize the CTC beam search decoder
    beam_search_decoder = cuda_ctc_decoder(
        tokens = vocab,
        nbest = 5,
        beam_size = 10
    )
    beam_search_decoder_cpu = ctc_decoder(
        lexicon = None,
        tokens = vocab,
        nbest = 5,
        beam_size = 10,
        blank_token = 'b',
        sil_token = 'b'
    )
    log_probs_cuda = log_probs.to('cuda')
    seq_len_cuda = seq_len.to('cuda')
    start = time.time()
    results_torch, tokens = beam_search(beam_search_decoder, log_probs_cuda, seq_len_cuda, return_paths = False)
    print("Elapsed time cuda beam search:",time.time()-start)
    start = time.time()
    results_ont,paths = fast_beam_search(logits)
    print("Elapsed time fast beam search:",time.time()-start)
    start = time.time()
    results_cpu, tokens_cpu = beam_search(beam_search_decoder_cpu, log_probs, seq_len, return_paths = False)
    print("Elapsed time cpu beam search:",time.time()-start)