# Some utility functions for the project

import torch


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
