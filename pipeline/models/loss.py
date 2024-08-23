import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSE(nn.Module):
    """
    Class to compute the Root Mean Squared Error as a PyTorch module.
    """

    def __init__(self):
        """
        Initialize the RMSE module.
        """
        super(RMSE, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Forward pass for computing RMSE between predictions and true values.

        Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.

        Returns:
        torch.Tensor: Computed RMSE value.
        """
        mse = F.mse_loss(y_pred, y_true, reduction="mean")
        return torch.sqrt(mse)
