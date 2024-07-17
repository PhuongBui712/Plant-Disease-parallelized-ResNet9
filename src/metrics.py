import torch
from torch import Tensor


def accuracy(y_pred: Tensor, y: Tensor):
    """
    Calculates the accuracy of a model's predictions.

    Args:
        y_pred (torch.Tensor): The model's unnormalized logits.
        y (torch.Tensor): The true labels.

    Returns:
        float: The accuracy of the model.
    """
    y_pred = torch.argmax(y_pred, 1)

    return (y_pred == y).type(torch.float).mean().item()