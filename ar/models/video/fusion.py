import torch

def SCI_weights(probabilities: torch.Tensor,
                log_probs: bool = False,
                from_logits: bool = False) -> torch.Tensor:
    """
    Computes the importance of a probability distribution. As lower the 
    entropy is, the larger the importance is

    Parameters
    ----------
    probabilities: torch.Tensor
        tensor containing the probabilities distribution. The tensor shape is
        (*, N) where N is the number of classes. The method automatically 
        broadcast to tensors with more dimensions
    log_probs: bool, default False
        Are the given probabilites log probs?
    from_logits: bool, default False
        Does the given probabilites are logits?
    
    Returns
    -------
    torch.Tensor
        Tensor of shape (*, C) containing the 'importance' 
        for each prob distribution in video M
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
    _, maxes = p.max(-1)

    return (N * maxes - 1) / (N - 1)


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
    _, maxes = p.max(-1, keepdim=True)

    # SCIs: (M, C)
    SCIs = (N * maxes - 1) / (N - 1)

    # pi: (M, N_CLASSES)
    pi = (SCIs * p).sum(1) 
    pi = pi / SCIs.sum(1)

    # (N_CLASSES,)
    return pi.max(0)[1]

