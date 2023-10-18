# ========================================
# FileName: functions.py
# Date: 18 juillet 2023 - 16:39
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Functions relative to SPDnet 
#        implementation
# =========================================

from typing import Tuple, Callable

import torch
from torch.autograd import Function

from .utils import symmetrize, construct_eigdiff_matrix, zero_offdiag


# =============================================================================
# BiMap
# =============================================================================
def biMap(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """ BiMap transform in a SPDnet layer according to the paper:

        \"A Riemannian Network for SPD Matrix Learning\", Huang et al
        AAAI Conference on Artificial Intelligence, 2017

    The mapping is as follows:

    .. math::

        \\mathbf{Y} = \\mathbf{W}^T \\mathbf{X} \\mathbf{W},

    where :math:`\\mathbf{X}\\in\\mathcal{S}_{n_{\\mathrm{in}}}^{++}` is the
    input SPD matrix and :math:`\\mathbf{W}\\in\\mathcal{S}t(n_{\\mathrm{in}},
    n_{\\mathrm{out}})` is an orthogonal weight matrix. For convenience, the
    input **W** of this function is the transpose of the weight matrix.

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_in, n_in)
        Batches of input SPD matrices.

    W : torch.Tensor of shape (..., n_out, n_in)
        Batch of Stiefel weight matrices
        (or their transpose in case n_out > n_in).
        W.ndim must be 2 if X.ndim is 2. Then for higher number of dimensions,
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
    assert X.shape[-1] == W.shape[-1], \
            "X and W must have the compatible dimensions."

    if X.ndim==2:
        assert W.ndim == 2, \
                f"W must be a 2-dimensional tensor for X.ndim={X.ndim}"
        _W = W.transpose(0, 1)
    else:
        assert X.ndim == W.ndim+1, \
                "X and W must have compatible dimensions: " +\
                f"X.ndim={X.ndim} and W.ndim={W.ndim}."
        # Repeat W in the -3 dimension to match the number of matrices in X
        _W = W.transpose(-1, -2).unsqueeze(-3).repeat(
                *[1 for _ in range(W.ndim-2)], X.shape[-3], 1, 1)

    return torch.einsum(
            '...cd,...de,...ef->...cf', torch.transpose(_W, -1, -2), X, _W)


def biMap_gradient(X: torch.Tensor, W: torch.Tensor,
                   grad_output: torch.Tensor) -> Tuple[torch.Tensor,
                                                       torch.Tensor]:
    """Gradient of biMap towars input and weight matrix

    Parameters
    ----------
    X : torch.Tensor of shape (..., n_in, n_in)
        Batches of input SPD matrices.

    W : torch.Tensor of shape (..., n_out, n_in)
        Batch of Stiefel weight matrices
        (or their transpose in case n_out > n_in).
        W.ndim must be 2 if X.ndim is 2. Then for higher number of dimesions,
        W.ndim must be X.ndim-1. (We will repeat the -3 dimension to match the
        number of matrices in X)

    grad_output: torch.Tensor of shape (..., n_out, n_out)
        Gradient of the loss with respect to the output of the layer.

    Returns
    -------
    grad_input : torch.Tensor of shape (..., n_in, n_in)
        Gradient of the loss with respect to the input of the layer.

    grad_weight : torch.Tensor of shape (..., n_out, n_in)
        Gradient of the loss with respect to the weight of the layer.
    """
    if X.ndim==2:
        _W = W.transpose(0, 1)
        grad_input = _W @ grad_output @ _W.transpose(0, 1)
        grad_weight = 2*grad_output @ W @ X

    else:
        _W = W.transpose(-1, -2).unsqueeze(-3).repeat(
                *[1 for _ in range(W.ndim-2)], X.shape[-3], 1, 1)

        grad_input = torch.einsum('...cd,...de,...ef->...cf', _W,
                               grad_output, _W.transpose(-1, -2))
        grad_weight = 2*torch.einsum('...bcd,...bde,...bef->...cf',
                                     grad_output, _W.transpose(-1, -2), X)

    return grad_input, grad_weight


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
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, 
                                                          torch.Tensor]:
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
        return biMap_gradient(X, W, grad_output)


# =============================================================================
# Operations  on the eigenvalues of a SPD matrix
# =============================================================================
def eig_operation(M: torch.Tensor, operation: Callable,
                  eig_function: str = "eigh", **kwargs) -> Tuple[
                          torch.Tensor, torch.Tensor,
                          torch.Tensor]:
    """Generic functions to compute an operation on the eigenvalues of a
    SPD matrix.

    Parameters
    ----------
    M : torch.Tensor
        SPD matrix of shape (..., n_features, n_features)

    operation: Callable
        function to apply to the eigenvalues. Any unknown keyword args passed
        to this function are passed to the operation function.

    eig_function: str
        name of the function to compute the eigenvalues and eigenvectors.
        Choices are: "eigh", "eig" or "svd"
        Default is "eigh" for torch.eigh.

    **kwargs:
        keyword arguments to pass to the operation function.

    Returns
    -------
    eigvals : torch.Tensor
        eigenvalues of shape (..., n_features)
        of the SPD matrices in M.

    eigvecs : torch.Tensor
        eigenvectors of shape (..., n_features, n_features)
        of the SPD matrices in M.

    result : torch.Tensor
        result of the operation on the eigenvalues of shape
        (..., n_matrices, n_features) and reconstructed from the
        eigenvectors.
    """
    # Parsing the eig_function argument
    assert eig_function in ["eigh", "eig", "svd"], \
            f"eig_function must be in ['eigh', 'eig', 'svd'], got {eig_function}"
    if eig_function == "eigh":
        _eig_function = torch.linalg.eigh
    elif eig_function == "eig":
        _eig_function = torch.linalg.eig
    else:
        _eig_function = torch.linalg.svd


    eigvals, eigvecs = _eig_function(M)
    _eigvals = torch.diag_embed(operation(eigvals, **kwargs))
    result = torch.einsum('...cd,...de,...ef->...cf',
                        eigvecs, _eigvals, eigvecs.transpose(-1, -2))

    return eigvals, eigvecs, result


def eig_operation_gradient(grad_output: torch.Tensor, eigvals: torch.Tensor,
                           eigvecs: torch.Tensor, operation: Callable,
                           grad_operation: Callable, **kwargs) -> torch.Tensor:
    """Generic function to compute the gradient of an operation on the
    eigenvalues of a SPD matrix.

    Parameters
    ----------
    grad_output : torch.Tensor
        gradient of the loss function wrt the output of the operation.
        of shape (..., n_features, n_features)

    eigvals : torch.Tensor
        eigenvalues of shape (..., n_features)

    eigvecs : torch.Tensor
        eigenvectors of shape (..., n_features, n_features)

    operation: Callable
        function to apply to the eigenvalues. Any unknown keyword args passed
        to this function are passed to the operation function.

    grad_operation: Callable
        function to apply to the gradient of the operation. Any unknown
        keyword args passed to this function are passed to the operation
        function.

    **kwargs:
        keyword arguments to pass to the operation and gradient functions.
    """
    grad_output_sym = symmetrize(grad_output)
    eigvals_ = torch.diag_embed(operation(eigvals, **kwargs))
    deriveigvals = torch.diag_embed(grad_operation(eigvals, **kwargs))
    eigdiff_matrix = construct_eigdiff_matrix(eigvals)
    eigvecs_transpose = eigvecs.transpose(-1, -2)
    grad_eigvectors = 2*torch.einsum(
            '...ab,...bc,...cd->...ad', grad_output_sym, eigvecs,
            eigvals_)
    grad_eigvals = torch.einsum(
            '...ab,...bc,...cd,...df->...af', deriveigvals,
            eigvecs_transpose, grad_output, eigvecs)

    # Computing final gradient towards X
    # -eigdiff_matrix cause transposing pairwise distances change the sign only
    return 2*torch.einsum('...ab,...bc,...cd->...ad',
                eigvecs,
                -eigdiff_matrix * symmetrize(
                    torch.einsum('...ab,...bc->...ac', eigvecs_transpose,
                                 grad_eigvectors)),
                eigvecs.transpose(-1, -2)) +\
            torch.einsum('...ab,...bc,...cd->...ad', eigvecs,
                zero_offdiag(grad_eigvals), eigvecs_transpose)


class ReEig(Function):
    """ReEig function."""

    @staticmethod
    def forward(ctx, M: torch.Tensor, eps: float) -> torch.Tensor:
        """Forward pass of the ReEig function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass.

        M : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices.

        eps : float
            Value for the rectification of the eigenvalues.

        Returns
        -------
        M_rect : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues.
        """
        operation = lambda x: torch.nn.functional.threshold(x, eps, eps)
        eigvals, eigvecs, M_rect = eig_operation(M, operation)
        ctx.save_for_backward(eigvals, eigvecs, eps)
        return M_rect

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass of the ReEig function.
        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to retrieve tensors saved during the forward pass.
        grad_output : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        grad_input : torch.Tensor of shape (..., n_features, n_features)
            Gradient of the loss with respect to the input of the layer.
        """
        operation = lambda x: torch.nn.functional.threshold(x, eps, eps)
        grad_operation = lambda x: (x > eps).double()
        eigvals, eigvecs, eps = ctx.saved_tensors
        return eig_operation_gradient(grad_output, eigvals, eigvecs,
                                operation, grad_operation)
