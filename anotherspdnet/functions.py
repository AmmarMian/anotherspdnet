# ========================================
# FileName: functions.py
# Date: 18 juillet 2023 - 16:39
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Functions relative to SPDnet 
#        implementation
# =========================================

from typing import Tuple

import torch
from torch.autograd import Function


# =============================================================================
# BiMap
# =============================================================================
def biMap(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """ BiMap transform in a SPDnet layer according to the paper
    A Riemannian Network for SPD Matrix Learning, Huang et al
    AAAI Conference on Artificial Intelligence, 2017

    The mapping is as follows: $\\mathbf{Y} = \\mathbf{W}^T \\mathbf{X} \\mathbf{W}$,

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_in, n_in)
        Batches of input SPD matrices.

    W : torch.Tensor of shape (..., n_out, n_in)
        Batch of Stiefel weight matrices (or their transpose in case n_out > n_in).
        W.ndim must be 2 if X.ndim is 2. Then for higher number of dimesions,
        W.ndim must be X.ndim-1. (We will repeat the -3 dimension to match the
        number of matrices in X)

    Returns
    -------
    Y : torch.Tensor of shape (..., n_out, n_out)
    """

    # Check the input
    assert X.ndim >= 2, "X must be at least a 2-dimensional tensor."
    assert W.ndim >= 2, "W must be a at least a 2-dimensional tensor."
    assert X.shape[-1] == X.shape[-2], "X must be square."
    assert X.shape[-1] == W.shape[-1], "X and W must have the compatible dimensions."

    if X.ndim==2:
        assert W.ndim == 2, f"W must be a 2-dimensional tensor for X.ndim={X.ndim}"
        _W = W.transpose(1, 2)
    else:
        assert X.ndim == W.ndim+1, "X and W must have compatible dimensions: " +\
                f"X.ndim={X.ndim} and W.ndim={W.ndim}."
        # Repeat W in the -3 dimension to match the number of matrices in X
        _W = W.transpose(-1, -2).unsqueeze(-3).repeat(
                *[1 for _ in range(W.ndim-2)], X.shape[-3], 1, 1)

    return torch.einsum(
            '...cd,...de,...ef->...cf', torch.transpose(_W, -1, -2), X, _W)


# Defining torch functional with the actual gradient computation
class BiMapFunction(Function):
    """Bilinear mapping function."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Forward pass of the bilinear mapping function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass.
        X : torch.Tensor of shape (n_bathces, n_matrices, n_in, n_in)
            Batch of several SPD matrices.
        W : torch.Tensor of shape (n_batches, n_in, n_out)
            Batch of Stiefel weight matrices.

        Returns
        -------
        Y : torch.Tensor of shape (n_batches, n_matrices, n_out, n_out)
        """
        Y = biMap(X, W)
        ctx.save_for_backward(X, W)
        return Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass of the bilinear mapping function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass.
        grad_output : torch.Tensor of shape (n_batches, n_matrices, n_out, n_out)
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input : torch.Tensor of shape (n_batches, n_matrices, n_in, n_in)
            Gradient of the loss with respect to the input of the layer.
        grad_weight : torch.Tensor of shape (n_batches, n_in, n_out)
            Gradient of the loss with respect to the weight of the layer.
        """
        X, W = ctx.saved_tensors

        if X.ndim==2:
            _W = W.transpose(1, 2)
        else:
            _W = W.transpose(-1, -2).unsqueeze(-3).repeat(
                    *[1 for _ in range(W.ndim-2)], X.shape[-3], 1, 1)

        grad_input = torch.einsum('...cd,...de,...ef->...cf', _W,
                               grad_output, _W.transpose(-1, -2))
        grad_weight = 2*torch.einsum('...bcd,...bde,...bef->...cf', grad_output,
                                _W.transpose(-1, -2), X)

        return grad_input, grad_weight

