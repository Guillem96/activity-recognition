import torch


def SCI_fusion(probabilities: torch.Tensor, 
               log_probs: bool = False,
               from_logits: bool = False) -> torch.Tensor:
    """
    Grouping predictions using an average, assumes that each clip has the same 
    importance. Usually, this is not true. SCI computes a weighted average given
    more importance to the predictions that have low entropy.

    Parameters
    ----------
    probabilities: torch.Tensor
        Tensor where each element contains the stacked probabilites of the 
        different clip's crops of shape (M, C, N_CLASSES), 
        where M is the number of clips and C is the number of crops of each clip.
        If you sampling strategy does not make different different crops for 
        each clip, you can feed to the function a tensor of shape (M, 1, N)
    log_probs: bool, default False
        Are the given probabilites log probs?
    from_logits: bool, default False
        Does the given probabilites are logits?
    
    Returns
    -------
    torch.Tensor
        Tensor of shape (N_CLASSES,) containing the score of each class.
    """

    if log_probs and from_logits:
        raise ValueError('log_probs and from_logits cannot be set to true'
                         ' at de same time.')
    
    if log_probs:
        p = probabilities.exp()
    elif from_logits:
        p = probabilities.softmax(-1)
    else:
        p = probabilities
    
    N = p.size(-1)
    maxes = p.max(-1, keepdim=True).values

    # SCIs: (M, C)
    SCIs = (N * maxes - 1) / (N - 1)

    print(SCIs.size())
    # pi: (M, N_CLASSES)
    pi = (SCIs * p).sum(1) 
    pi = pi / SCIs.sum(1)

    # (N_CLASSES,)
    return pi.max(0).values

