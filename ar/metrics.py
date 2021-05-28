import functools
from typing import Any

import torch


def top_k_accuracy(y_preds: torch.Tensor,
                   y_trues: torch.Tensor,
                   k: int = 5) -> torch.Tensor:
    """
    Computes the top k accuracy

    Parameters
    ----------
    y_preds: torch.Tensor of shape [BATCH, N_CLASSES]
        Log probabilities, probabilities or logits from the model
    y_trues: torch.LongTensor of shape [BATCH]
        Ground truths. With range [0, N_CLASSES]
    k: int, default 5
        Top k accuracy to compute

    Returns
    -------
    torch.Tensor
    """
    bs = y_preds.size(0)
    y_trues = y_trues.view(-1)
    assert bs == y_trues.size(0)

    _, preds_k = y_preds.topk(k, dim=-1)
    y_trues = y_trues.unsqueeze(1).repeat(1, k)

    return preds_k.eq(y_trues).any(-1).sum().float() / y_preds.size(0)


def top_5_accuracy(y_preds: torch.Tensor,
                   y_trues: torch.Tensor) -> torch.Tensor:
    return top_k_accuracy(y_preds, y_trues, k=5)


def top_3_accuracy(y_preds: torch.Tensor,
                   y_trues: torch.Tensor) -> torch.Tensor:
    return top_k_accuracy(y_preds, y_trues, k=3)


def accuracy(y_preds: torch.Tensor, y_trues: torch.Tensor) -> torch.Tensor:
    return top_k_accuracy(y_preds, y_trues, k=1)
