# Some utility functions for the project

from typing import Optional

import math
import torch
from geoopt.tensor import ManifoldParameter


def initialize_weights_stiefel(W: ManifoldParameter,
                               seed: Optional[int] = None) -> ManifoldParameter:
    """Initialize the weights as being on the Stiefel manifold.

    Theorem 2.2.1 in Chikuse (2003): statistics on special manifolds.
    TODO: Verify that this is correct.

    Parameters
    ----------
    W: geoopt.tensor.ManifoldParameter of shape (..., n, k)
        weights to be initialized. n should be greater than k.

    seed: int, optional
        random seed for reproducibility. If None, no seed is used.

    Returns
    -------
    W: geoopt.tensor.ManifoldParameter
        initialized weights (same object as input with data changed)
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    _W = torch.randn(W.shape, dtype=W.dtype, device=W.device,
                    generator=generator)
    temp = torch.einsum('...ij,...jk->...ik', _W.transpose(-1, -2), _W)
    eigvals, eigvec = torch.linalg.eig(temp)
    eigvals = torch.diag_embed(1/torch.sqrt(eigvals.real))
    invsqrt_W = torch.einsum('...ij,...jk,...kl->...il', eigvec.real, eigvals,
                            eigvec.real.transpose(-1, -2))
    _W = torch.einsum('...ij,...jk->...ik', _W, invsqrt_W)
    W.data = _W
    return W


def initialize_weights_sphere(W: ManifoldParameter,
                              seed: Optional[int] = None) -> ManifoldParameter:
    """Initialize the weights as being on the sphere manifold.

    Parameters
    ----------
    W: geoopt.tensor.ManifoldParameter of shape (..., n, k)
        weights to be initialized

    seed: int, optional
        random seed for reproducibility. If None, no seed is used.

    Returns
    -------
    W: geoopt.tensor.ManifoldParameter
        initialized weights (same object as input with data changed)
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    # From TSMNet
    # kaiming initialization std2uniformbound * gain * fan_in
    bound = math.sqrt(3) * 1. / W.shape[-1]
    _W = 2*torch.rand(W.shape, dtype=W.dtype, device=W.device,
                      generator=generator)*bound - bound
    # Satisfying constraint
    W.data = _W / torch.norm(_W, dim=-1, keepdim=True)
    return W



def symmetrize(X: torch.Tensor) -> torch.Tensor:
    """Symmetrize a tensor along the last two dimensions.

    Parameters
    ----------
    X : torch.Tensor
        tensor of shape (..., n_features, n_features)

    Returns
    -------
    sym_X : torch.Tensor
        symmetrized tensor of shape (..., n_features, n_features)
    """
    return .5*(X + X.transpose(-1, -2))


def construct_eigdiff_matrix(eigvals: torch.Tensor) -> torch.Tensor:
    """Constructs the matrix of the inverse pairwise differences between
    eigenvalues on the off-diagonal and 0 on the diagonal.

    Parameters
    ----------
    eigvals : torch.Tensor of shape (..., n_features)
        eigenvalues of the SPD matrices

    Returns
    -------
    eigdiff_matrix : torch.Tensor of shape (..., n_features, n_features)
        matrix of the inverse pairwise differences between eigenvalues
    """
    eigdiff_matrix = 1/(eigvals.unsqueeze(-1) - eigvals.unsqueeze(-2))
    eigdiff_matrix[torch.isinf(eigdiff_matrix)] = 0.
    return eigdiff_matrix


def zero_offdiag(X: torch.Tensor) -> torch.Tensor:
    """Sets the off-diagonal elements of a tensor to 0.

    Parameters
    ----------
    X : torch.Tensor
        tensor of shape (..., n_features, n_features)

    Returns
    -------
    X_zero : torch.Tensor
        tensor of shape (..., n_features, n_features) with 0 on the
        off-diagonal
    """
    return torch.diag_embed(torch.diagonal(X, dim1=-2, dim2=-1))
