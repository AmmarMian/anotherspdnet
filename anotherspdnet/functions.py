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

from math import prod
import torch
from torch.autograd import Function

from .utils import (
        symmetrize, construct_eigdiff_matrix, zero_offdiag,
        nd_tensor_to_3d, threed_tensor_to_nd)


# =============================================================================
# BiMap
# =============================================================================
def biMap(X: torch.Tensor, W: torch.Tensor,
          mode: str = "einsum") -> torch.Tensor:
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

    mode : str
        Mode for the computation of the bilinear mapping. Choices are:
            "einsum" or "bmm". Default is "einsum".

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
    assert mode in ["einsum", "bmm"], \
            f"mode must be in ['einsum', 'bmm'], got {mode}"

    if X.ndim==2:
        assert W.ndim == 2, \
                f"W must be a 2-dimensional tensor for X.ndim={X.ndim}"
        if mode == "einsum":
            return torch.einsum('ij,jk,kl->il', W, X, W.T)
        else:
            return W @ X @ W.transpose(0, 1)
    else:
        assert X.ndim == W.ndim+1, \
                "X and W must have compatible dimensions: " +\
                f"X.ndim={X.ndim} and W.ndim={W.ndim}."
        if mode == "einsum":
            return torch.einsum(
            '...cd,...ide,...ef->...icf', W, X, W.transpose(-1, -2))

        else:

            # We need to construct 3-D tensors with the same number of matrices
            _X = nd_tensor_to_3d(X)
            
            n_repeats = X.shape[-3]
            repeat_tuple = (1,)*(W.ndim-2) + (n_repeats,) + (1,)*2
            _W = W.unsqueeze(-3).repeat(*repeat_tuple)
            _W = nd_tensor_to_3d(_W)

            result = torch.bmm(_W, torch.bmm(_X, _W.transpose(-1, -2)))
            n_out = W.shape[-2]
            return threed_tensor_to_nd(result, X.shape[:-2] + (n_out, n_out))


def biMap_gradient(X: torch.Tensor, W: torch.Tensor,
                   grad_output: torch.Tensor,
                   mode: str = "einsum") -> Tuple[torch.Tensor,
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

    mode : str
        Mode for the computation of the bilinear mapping. Choices are:
            "einsum" or "bmm". Default is "einsum".

    Returns
    -------
    grad_input : torch.Tensor of shape (..., n_in, n_in)
        Gradient of the loss with respect to the input of the layer.

    grad_weight : torch.Tensor of shape (..., n_out, n_in)
        Gradient of the loss with respect to the weight of the layer.
    """
    if X.ndim==2:

        if mode == "einsum":
            grad_input = torch.einsum('ij,jk,kl->il', W.T, grad_output, W)
            grad_weight = 2*torch.einsum('ij,jk,kl->il', grad_output, W, X)
        else:
            grad_input = W.transpose(0, 1) @ grad_output @ W
            grad_weight = 2*grad_output @ W @ X

    else:

        if mode == "einsum":
            grad_input = torch.einsum('...ab,...ibc,...cd->...iad',
                                      W.transpose(-1, -2), grad_output, W) 

            grad_weight = 2*torch.einsum('...iab,...bc,...icd->...ad',
                                         grad_output, W, X)

        else:

            _X = nd_tensor_to_3d(X)
            repeat_tuple = (1,)*(W.ndim-2) + (X.shape[-3],) + (1,)*2
            _W = W.unsqueeze(-3).repeat(*repeat_tuple)
            _W = nd_tensor_to_3d(_W)

            grad_input = torch.bmm(_W.transpose(-1, -2),
                                torch.bmm(grad_output, _W))
            grad_input = threed_tensor_to_nd(grad_input, X.shape)
            grad_weight = 2*torch.bmm(grad_output, torch.bmm(_W, _X))
            grad_weight = threed_tensor_to_nd(grad_weight, W.shape)

    return grad_input, grad_weight


# Defining torch functional with the actual gradient computation
class BiMapFunction(Function):
    """Bilinear mapping function."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor,
                mode: str = "einsum") -> torch.Tensor:
        """Forward pass of the bilinear mapping function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass.
        X : torch.Tensor of shape (n_bathces, n_matrices, n_in, n_in)
            Batch of several SPD matrices.

        W : torch.Tensor of shape (n_batches, n_in, n_out)
            Batch of Stiefel weight matrices.

        mode : str
            Mode for the computation of the bilinear mapping. Choices are:
                "einsum" or "bmm". Default is "einsum".

        Returns
        -------
        Y : torch.Tensor of shape (n_batches, n_matrices, n_out, n_out)
        """
        Y = biMap(X, W, mode=mode)
        ctx.mode = mode
        ctx.save_for_backward(X, W)
        return Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, 
                                                          torch.Tensor, None]:
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
        mode = ctx.mode
        return biMap_gradient(X, W, grad_output, mode=mode) + (None,)


# =============================================================================
# Operations  on the eigenvalues of a SPD matrix
# =============================================================================
def eig_operation(M: torch.Tensor, operation: Callable,
                  eig_function: str = "eigh", 
                  mm_mode: str = "einsum",
                  **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Name of the function to compute the eigenvalues and eigenvectors.
        Choices are: "eigh" or "eig".
        Default is "eig" for torch.eig.

    mm_mode : str
        Mode for the computation of the matrix multiplication. Choices are:
            "einsum" or "bmm". Default is "einsum".

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
    assert eig_function in ["eigh", "eig"], \
            f"eig_function must be in ['eigh', 'eig'], got {eig_function}"
    if eig_function == "eigh":
        _eig_function = torch.linalg.eigh
    else:
        _eig_function = torch.linalg.eig


    eigvals, eigvecs = _eig_function(M)
    eigvals, eigvecs = torch.abs(eigvals), torch.real(eigvecs)
    _eigvals = torch.diag_embed(operation(eigvals, **kwargs))
    if mm_mode == "einsum":
        result = torch.einsum('...cd,...de,...ef->...cf',
                            eigvecs, _eigvals, eigvecs.transpose(-1, -2))
    else:
        result = eigvecs @ _eigvals @ eigvecs.transpose(-1, -2)

    return eigvals, eigvecs, result


def eig_operation_gradient_eigs(grad_output: torch.Tensor, eigvals: torch.Tensor,
                eigvecs: torch.Tensor, operation: Callable,
                grad_operation: Callable, mm_mode: str = "einsum",
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradient of an operation on the eigenvalues of a SPD matrix towards the
    eigenvalues and eigenvectors.

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

    mm_mode : str
        Mode for the computation of the matrix multiplication. Choices are:
            "einsum" or "bmm". Default is "einsum".

    **kwargs:
        keyword arguments to pass to the operation and gradient functions.

    Returns
    --------
    grad_eigvals : torch.Tensor
        gradient of the loss with respect to the eigenvalues of the layer.

    grad_eigvecs : torch.Tensor
        gradient of the loss with respect to the eigenvectors of the layer.
    """
    grad_output_sym = symmetrize(grad_output)
    eigvals_ = torch.diag_embed(operation(eigvals, **kwargs))
    deriveigvals = torch.diag_embed(grad_operation(eigvals, **kwargs))

    if mm_mode == "einsum":
        grad_eigvectors = 2*torch.einsum(
                '...ab,...bc,...cd->...ad', grad_output_sym, eigvecs,
                eigvals_)
        grad_eigvals = torch.einsum(
                '...ab,...bc,...cd,...df->...af', deriveigvals,
                eigvecs.transpose(-1, -2), grad_output_sym, eigvecs)

    else:
        # grad_eigvectors = 2*torch.bmm(
        #         grad_output_sym, torch.bmm(eigvecs, eigvals_))
        # grad_eigvals = torch.bmm(
        #     torch.bmm(deriveigvals, eigvecs.transpose(-1, -2)),
        #     torch.bmm(grad_output_sym, eigvecs))
        grad_eigvectors = 2*(grad_output_sym @ eigvecs @ eigvals_)
        grad_eigvals = deriveigvals @ eigvecs.transpose(-1, -2) @\
                grad_output_sym @ eigvecs

    return zero_offdiag(grad_eigvals), grad_eigvectors



def eig_operation_gradient(grad_output: torch.Tensor, eigvals: torch.Tensor,
                           eigvecs: torch.Tensor, operation: Callable,
                        grad_operation: Callable, mm_mode: str = "einsum",
                           **kwargs) -> torch.Tensor:
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

    mm_mode : str
        Mode for the computation of the matrix multiplication. Choices are:
            "einsum" or "bmm". Default is "einsum".

    **kwargs:
        keyword arguments to pass to the operation and gradient functions.

    Returns
    ---------
    grad_input : torch.Tensor
        gradient of the loss with respect to the input of the layer.
    """
    eigdiff_matrix = construct_eigdiff_matrix(eigvals)
    eigvecs_transpose = eigvecs.transpose(-1, -2)
    grad_eigvals, grad_eigvectors = eig_operation_gradient_eigs(
            grad_output, eigvals, eigvecs, operation, grad_operation, 
            mm_mode, **kwargs)

    # Computing final gradient towards X
    if mm_mode == "einsum":
        grad_input = 2*torch.einsum('...ab,...bc,...cd->...ad',
                    eigvecs,
                    eigdiff_matrix.transpose(-1, -2) * symmetrize(
                        torch.einsum('...ab,...bc->...ac', eigvecs_transpose,
                                     grad_eigvectors)),
                    eigvecs_transpose) +\
                torch.einsum('...ab,...bc,...cd->...ad', eigvecs,
                    zero_offdiag(grad_eigvals), eigvecs_transpose)

    else:
        # grad_input = 2*torch.bmm(
        #         eigvecs, torch.bmm(
        #             eigdiff_matrix.transpose(-1, -2) * symmetrize(
        #                 torch.bmm(eigvecs_transpose, grad_eigvectors)),
        #         eigvecs_transpose)) +\
        #         torch.bmm(eigvecs, torch.bmm(
        #             zero_offdiag(grad_eigvals), eigvecs_transpose))
        grad_input = 2*(eigvecs @ (eigdiff_matrix.transpose(-1, -2) *\
                        symmetrize(eigvecs_transpose @ grad_eigvectors)) @
                        eigvecs_transpose) +\
                     eigvecs @ (zero_offdiag(grad_eigvals)) @ eigvecs_transpose


    return grad_input


# ReEig
# -----------------------------
def re_operation(eigvals: torch.Tensor, eps: float) -> torch.Tensor:
    """Rectification of the eigenvalues of a SPD matrix.

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        eigenvalues of the SPD matrices

    eps : float
        Value for the rectification of the eigenvalues.

    Returns
    -------
    eigvals_rect : torch.Tensor of shape (..., n_features)
        eigenvalues of the SPD matrices with rectified eigenvalues.
    """
    return torch.nn.functional.threshold(eigvals, eps, eps)


def re_operation_gradient(eigvals: torch.Tensor, eps: float,
                          dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Gradient of the rectification of the eigenvalues of a SPD matrix.

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        eigenvalues of the SPD matrices

    eps : float
        Value for the rectification of the eigenvalues.

    dtype : Callable
        Casting type of the gradient. Default is torch.float64.

    Returns
    -------
    eigvals_rect : torch.Tensor of shape (..., n_features)
        eigenvalues of the SPD matrices with rectified eigenvalues.
    """
    return (eigvals > eps).type(dtype)


class ReEigFunction(Function):
    """ReEig function."""

    @staticmethod
    def forward(ctx, M: torch.Tensor, eps: float,
                mm_mode: str = "einsum",
                eig_function: str = "eigh") -> torch.Tensor:
        """Forward pass of the ReEig function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass.

        M : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices.

        eps : float
            Value for the rectification of the eigenvalues.

        mm_mode : str
            Mode for the computation of the matrix multiplication. Choices are:
                "einsum" or "bmm". Default is "einsum".

        eig_function: str
            Name of the function to compute the eigenvalues and eigenvectors.
            Choices are: "eigh" or "eig".

        Returns
        -------
        M_rect : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues.
        """
        operation = lambda x: re_operation(x, eps)
        eigvals, eigvecs, M_rect = eig_operation(M, operation, eig_function,
                                                 mm_mode)
        ctx.save_for_backward(eigvals, eigvecs)
        ctx.eps = eps
        ctx.mm_mode = mm_mode
        return M_rect

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) ->\
            Tuple[torch.Tensor, None, None, None]:
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
        eps = ctx.eps
        mm_mode = ctx.mm_mode
        operation = lambda x: re_operation(x, eps)
        operation_gradient = lambda x: re_operation_gradient(x, eps,
                                                             grad_output.dtype)
        eigvals, eigvecs = ctx.saved_tensors
        return eig_operation_gradient(grad_output, eigvals, eigvecs,
                                operation, operation_gradient, mm_mode),\
                None, None, None


# LogEig
# -----------------------------
class LogEigFunction(Function):
    """LogEig function."""

    @staticmethod
    def forward(ctx, M: torch.Tensor,
                mm_mode: str = "einsum",
                eig_function: str = "eigh") -> torch.Tensor:
        """Forward pass of the logEig function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save tensors for the backward pass.

        M : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices.

        mm_mode : str
            Mode for the computation of the matrix multiplication. Choices are:
                "einsum" or "bmm". Default is "einsum".

        eig_function: str
            Name of the function to compute the eigenvalues and eigenvectors.
            Choices are: "eigh" or "eig".

        Returns
        -------
        M_rect : torch.Tensor of shape (..., n_features, n_features)
            Batch of SPD matrices with rectified eigenvalues.
        """
        eigvals, eigvecs, M_rect = eig_operation(M, torch.log, eig_function,
                                                 mm_mode)
        ctx.mm_mode = mm_mode
        ctx.save_for_backward(eigvals, eigvecs)
        return M_rect

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) ->\
            Tuple[torch.Tensor, None, None]:
        """Backward pass of the logEig function.

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
        mm_mode = ctx.mm_mode
        operation_gradient = lambda x: 1/x
        eigvals, eigvecs = ctx.saved_tensors
        return eig_operation_gradient(grad_output, eigvals, eigvecs,
                                torch.log, operation_gradient, mm_mode),\
                None, None


# =============================================================================
# Vectorisation stuff
# =============================================================================
def vec_batch(X: torch.Tensor) -> torch.Tensor:
    """Vectorize a batch of tensors along last two dimensions.

    Parameters
    ----------
    X : torch.Tensor of shape (..., n, k)
        Batch of matrices.

    Returns
    -------
    X_vec : torch.Tensor of shape (..., n*k)
        Batch of vectorized matrices.
    """
    return X.reshape(*X.shape[:-2], -1)

def unvec_batch(X_vec: torch.Tensor, n: int) -> torch.Tensor:
    """Unvectorize a batch of tensors along last dimension.
    Parameters
    ----------
    X_vec : torch.Tensor of shape (..., n*k)
        Batch of vectorized matrices.

    n : int
        Number of rows of the matrices.
    Returns
    -------
    X : torch.Tensor of shape (..., n, k)
        Batch of matrices.
    """
    return X_vec.reshape(*X_vec.shape[:-1], n, -1)


def vech_batch(X: torch.Tensor) -> torch.Tensor:
    """Vectorize the lower triangular part of a batch of square matrices.

    Parameters
    ----------
    X : torch.Tensor of shape (..., n, n)
        Batch of matrices.

    Returns
    -------
    X_vech : torch.Tensor of shape (..., n*(n+1)//2)
        Batch of vectorized matrices.
    """
    indices = torch.tril_indices(*X.shape[-2:])
    return X[..., indices[0], indices[1]]


def unvech_batch(X_vech: torch.Tensor) -> torch.Tensor:
    """Unvectorize a batch of tensors along last dimension.
    Parameters
    ----------
    X_vech : torch.Tensor of shape (..., n*(n+1)//2)
        Batch of vectorized matrices.
    Returns
    -------
    X : torch.Tensor of shape (..., n, n)
        Batch of matrices.
    """
    n = 0.5 * (-1 + torch.sqrt(torch.Tensor([1 + 8 * X_vech.shape[-1]])))
    n = int(torch.round(n))
    indices = torch.tril_indices(n, n)
    X = torch.zeros(*X_vech.shape[:-1], n, n, dtype=X_vech.dtype,
                    device=X_vech.device)
    X[..., indices[0], indices[1]] = X_vech
    X = symmetrize(X)
    return X


