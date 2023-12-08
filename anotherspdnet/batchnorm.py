#  Stuff related to batch normalization

from typing import List, Optional, Tuple

import torch
from torch import nn
from .functions import (
        SqrtmEigFunction, InvSqrtmEigFunction, 
        LogEigFunction, ExpEigFunction,
        spd_affine_invariant_geodesic
    )


from geoopt.manifolds import SymmetricPositiveDefinite
from geoopt.tensor import ManifoldParameter
import warnings


# from pyriemann.utils.mean import mean_riemann

# Riemannian mea mean
# TODO : compute the mean along axes we want
# ---------------
def riemannian_mean_spd(X: torch.Tensor, initial_stepsize: float = 1.0, 
                     max_iter: int = 5, tol: float = 1e-6, 
                     weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Riemannian mean for SPD matrices.

    Inspired by pyRiemann's implementation.

    Parameters
    ----------
    X : torch.Tensor of shape (..., N, N)
        Batch of SPD matrices. The mean is computed along the ... axes.

    initial_stepsize : float
        Initial step size to move on the tangent space.

    max_iter : int
        Maximum number of iterations.

    tol : float
        Tolerance to stop the algorithm.

    weights : torch.Tensor
        Weights to apply to the samples. Must be of same shape as X until
        ndim - 2. The weights are applied on the ... axes. by default, all are
        put to one.

    Returns
    -------
    mean : torch.Tensor of shape (N, N)
        Riemannian mean.
    """

    if weights is None:
        weights = torch.ones(X.shape[:-2], dtype=X.dtype, device=X.device)
    else:
        assert weights.shape == X.shape[:len(X.shape) - 2]
        assert torch.all(weights >= 0), "Weights must be positive"
    weights = weights / torch.sum(weights)

    with torch.no_grad():
        
        # Initialisation
        mean = torch.einsum('...ij,...->ij', X, weights)
        nu = initial_stepsize
        tau = torch.finfo(X.dtype).max
        crit = torch.finfo(X.dtype).max

        # Loop
        for _ in range(max_iter):

            # Compute useful matrices in tangent space
            eigvals, eigvecs = torch.linalg.eigh(mean)
            sqrt_eigvals = torch.sqrt(eigvals)
            inv_sqrt_eigvals = 1/sqrt_eigvals
            C12 = eigvecs @ torch.diag_embed(sqrt_eigvals) @ \
                    eigvecs.transpose(-2, -1)
            C12_inv = eigvecs @ torch.diag(inv_sqrt_eigvals) @ \
                    eigvecs.transpose(-2, -1)

            # Getting the tangent space matrix
            eigvals, eigvecs = torch.linalg.eigh(C12_inv @ X @ C12_inv)
            log_eigvals = torch.log(eigvals)
            log_matrices = torch.einsum('...ij,...jk,...kl->...il',
                                        eigvecs,
                                        torch.diag_embed(log_eigvals),
                                        eigvecs.transpose(-2, -1))
            J = torch.einsum('...,...ij->ij', weights, log_matrices)

            # Getting the new mean on the manifold
            eigvals, eigvecs = torch.linalg.eigh(nu*J)
            exp_eigvals = torch.exp(eigvals)
            exp_J = eigvecs @ torch.diag(exp_eigvals) @ \
                    eigvecs.transpose(-2, -1)
            mean = C12 @ exp_J @ C12

            # Check for convergence
            crit = torch.linalg.norm(J, ord='fro')
            h = nu*crit
            if h < tau:
                nu = 0.95*nu
                tau = h
            else:
                nu = 0.5*nu

            if crit <= tol or nu <= tol:
                break

        else:
            warnings.warn(f"Mean did not converge after {max_iter} iterations.")


    return mean


# Batch normalization
# ------------------
class BatchNormSPD(nn.Module):

    def __init__(self, n_features: int, momentum: float = 0.1,
                 tol_mean: float = 1e-6, max_iter_mean: int = 5,
                 initial_stepsize_mean: float = 1.0, 
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float64) -> None:
        """Batch normalization for SPD matrices. From paper:
        Brooks et al., “Riemannian Batch Normalization for SPD Neural Networks.”
        NEURIPS 2019.

        Parameters
        ----------
        n_features : int
            Number of features.

        momentum : float
            Momentum for the running covariance mean

        tol_mean : float
            Tolerance for the mean computation.

        max_iter_mean : int
            Maximum number of iterations for the mean computation.

        initial_stepsize_mean : float
            Initial step size for the mean computation.

        device : torch.device
            Device to use.
        """

        super().__init__()
        self.n_features = n_features
        self.momentum = momentum
        self.tol_mean = tol_mean
        self.max_iter_mean = max_iter_mean
        self.initial_stepsize_mean = initial_stepsize_mean
        self.manifold = SymmetricPositiveDefinite()
        self.device = device
        self.dtype = dtype
        self.running_mean = torch.eye(n_features, dtype=dtype, device=device)
        self.bias = ManifoldParameter(
                torch.eye(n_features, dtype=dtype, device=device),
                manifold=self.manifold, requires_grad=True)

    def forward(self, X: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor of shape (..., N, N)
            Batch of SPD matrices. The mean is computed along the ... axes.

        bias : torch.Tensor of shape (N, N)
            Bias.
        Returns
        -------
        Y : torch.Tensor of shape (..., N, N)
            Batch of SPD matrices.
        """

        if self.training:
            # Compute the mean of the batch
            new_mean = riemannian_mean_spd(
                    X, initial_stepsize=self.initial_stepsize_mean,
                    max_iter=self.max_iter_mean, tol=self.tol_mean)

            # TODO: Understand why we need to no grad here
            with torch.no_grad():
                self.running_mean = spd_affine_invariant_geodesic(
                        self.running_mean, new_mean, self.momentum)


        else:
            new_mean = self.running_mean

        isqrtm_new_mean = InvSqrtmEigFunction.apply(new_mean)
        sqrtm_bias = SqrtmEigFunction.apply(self.bias)
        X_centered = torch.einsum('ij,...jk,kl->...il',
                                isqrtm_new_mean, X, isqrtm_new_mean)
        X_normalized = torch.einsum('ij,...jk,kl->...il',
                                sqrtm_bias, X_centered, sqrtm_bias)

        return X_normalized


    def __repr__(self):
        return f"BatchNormSPD({self.n_features}, momentum={self.momentum}, "\
                f"tol_mean={self.tol_mean}, max_iter_mean={self.max_iter_mean}, "\
                f"initial_stepsize_mean={self.initial_stepsize_mean}, "\
                f"device={self.device}, dtype={self.dtype})"

    def __str__(self):
        return self.__repr__()

