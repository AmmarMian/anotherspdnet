# Estimation of covariance matrices

import warnings
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from tqdm import tqdm
from .functions import InvSqrtmEigFunction


def normalize_trace(Sigma_batch: torch.Tensor) -> torch.Tensor:
    """Normalize covariance by the trace (trace is equal to n_features).
    
    Parameters
    ----------
    Sigma_batch: torch.Tensor
        Batch of covariance matrices of shape (..., n_features, n_features),
        where `...` are the batches dimensions

    Returns
    -------
    torch.Tensor
        Normalized batch of covariance matrices of shape
        (..., n_features, n_features)

    """
    traces = torch.einsum('...ii->...', Sigma_batch).unsqueeze(-1).unsqueeze(-1)
    return Sigma_batch.shape[-2]*Sigma_batch/traces


def normalize_determinant(Sigma_batch: torch.Tensor) -> torch.Tensor:
    """Normalize covariance by the determinant (determinant=1).
    
    Parameters
    ----------
    Sigma_batch: torch.Tensor
        Batch of covariance matrices of shape (..., n_features, n_features),
        where `...` are the batches dimensions

    Returns
    -------
    torch.Tensor
        Normalized batch of covariance matrices of shape
        (..., n_features, n_features)

    """
    det = torch.linalg.det(Sigma_batch).unsqueeze(-1).unsqueeze(-1)
    return Sigma_batch/(torch.pow(det, 1/Sigma_batch.shape[-2]))


def student_function(
        x: torch.Tensor, n_features: int, nu: float) -> torch.Tensor:
    """Student function.
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    n_features : int
        Number of features.

    nu : float
        Degrees of freedom.
    Returns
    -------
    torch.Tensor
        Computed Student function over input tensor.
    """
    return (n_features + nu) / (nu +x)


def huber_function(
        x: torch.Tensor,
        delta: float,
        beta: float) -> torch.Tensor:
    """Huber function defined as:
        * u(x) = 1/beta is x <= delta
        * u(x) = delta/(beta*x) if x > delta


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    delta : float
        Threshold value.

    beta : float
        Scaling factor.

    Returns
    -------
    torch.Tensor
        Computed Huber function over input tensor.
    """
    return torch.where(
                x <= delta,
                1 / beta,
                delta / (beta * x)
            )


def tyler_function(x: torch.Tensor, n_features: int) -> torch.Tensor:
    """Tyler function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    n_features : int
        Number of features.

    Returns
    -------
    torch.Tensor
        Computed Tyler function over input tensor.
    """
    return n_features/x


class SCM(nn.Module):
    """Layer to compute SCM to estimate covariance matrix."""

    def __init__(self, correction: Optional[int] = 1,
                 assume_centered: Optional[bool] = False) -> None:
        super().__init__()
        self.correction = correction
        self.assume_centered = assume_centered

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute SCM over data.

        Parameters
        ----------
        X : torch.Tensor of shape (..., n_samples, n_features)
            Input tensor batch.

        Returns
        -------
        torch.Tensor of shape (..., n_features, n_features)
            Estimated covariance matrices (one per batch).
        """
        if self.assume_centered:
            _X = X
        else:
            _X = X - X.mean(dim=-2, keepdim=True)

        Sigma = torch.einsum('...ij,...jk->...ik',
                            _X.transpose(-2, -1),
                            _X)/(X.shape[-2]-self.correction)
        # Just to be sure
        return .5*(Sigma + Sigma.transpose(-2,-1))


class Mestimation(nn.Module):
    """Torch implementation of M-estimators of covariance matrix."""

    def __init__(
        self,
        m_estimation_function: Callable,
        n_iter: int = 30,
        tol: float = 1e-6,
        verbose: bool = False,
        assume_centered: bool = False,
        normalize: Optional[Callable] = None) -> None:
        """
        Initializes the M-estimation module.

        Parameters
        ----------
        m_estimation_function : Callable
            The M-estimation function to use.

        n_iter : int, optional (default=30)
            The number of iterations to perform.

        tol: float, optional (default=1e-6)
            Tolerance for stopping criterion.

        verbose : bool, optional (default=False)
            Whether to display a progress bar during estimation.

        assume_centered : bool, optional (default=False)
            Whether to assume that the data is already centered.

        normalize : Callable, optional (default=None)
            A function to normalize the covariance matrix.
            If None, no normalization will be performed.
        """
        super().__init__()
        self.m_estimation_function = m_estimation_function 
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.assume_centered = assume_centered
        self.normalize = normalize

    def _init_pbar(self) -> None:
        """Initialize progress bar.
        Parameters
        ----------
        n_iter : int
            Number of iterations.
        """
        if self.verbose:
            self.pbar = tqdm(total=self.n_iter, desc="M-estimation", leave=True)

    def _update_pbar(self, delta: float) -> None:
        """Update progress bar."""
        if self.verbose:
            self.pbar.set_postfix({"delta": f"{delta:.2e}"})
            self.pbar.update(1)

    def _iter_fixed_point(
        self, isqrtm_Sigma_prev: torch.Tensor, X: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One iteration of fixed point algorithm for M-estimation.

        Parameters
        ----------
        isqrtm_Sigma_prev : torch.Tensor of shape (..., n_features, n_features)
            Inverse square root of previous estimate of covariance matrices.
            where `...` are the batches dimensions.

        X : torch.Tensor of shape (..., n_samples, n_features)
            Input tensor, where `...` are the batches dimensions.

        **kwargs
            Additional keyword arguments to pass to the M-estimation function.

        Returns
        -------
        torch.Tensor of shape (..., n_features, n_features)
            Updated estimate of covariance matrices.

        torch.Tensor of shape (..., n_features, n_features)
            inverse sqrtm of updated estimate of covariance matrices.
        """
        batches_dimensions = X.shape[:-2]
        temp = torch.einsum('...ij,...jk->...ik',
                            isqrtm_Sigma_prev,
                            X.transpose(-2, -1))
        quadratic = self.m_estimation_function(
            torch.einsum("...ij,...ji->...i", temp.transpose(-2, -1), temp),
            **kwargs
        )
        temp = X.transpose(-2, -1) * torch.sqrt(
            quadratic.unsqueeze(-2).repeat(
                (1,)*len(batches_dimensions)+(X.shape[-1], 1))
        )
        Sigma = torch.einsum('...ij,...jk->...ik',
                             temp, temp.transpose(-2, -1)) / X.shape[-2]
        isqrtm_Sigma = InvSqrtmEigFunction.apply(Sigma)

        return Sigma, isqrtm_Sigma

    def forward(self, X: torch.Tensor,
                init: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Compute M-estimator of covariance matrix on a batch of data.

        Parameters
        ----------
        X : torch.Tensor of shape (..., n_samples, n_features)
            Input tensor, where `...` are the batches dimensions.

        init : torch.Tensor, optional (default=None)
            The initial estimate of the covariance matrix.
            If None, it will be initialized as the identity matrix.
            The shape should be (n_features, n_features) in which case, it is
            repeated over batches dimensions or it can be
            (..., n_features, n_features), where `...` are the batches dimensions.

        **kwargs
            Additional keyword arguments to pass to the M-estimation function.

        Returns
        -------
        torch.Tensor of shape (..., n_features, n_features)
            Estimated covariance matrices (one per batch).
        """
        batches_dimensions = X.shape[:-2]
        if not init:
            Sigma = torch.eye(X.shape[-1], device=X.device)
            for dim in reversed(batches_dimensions):
                Sigma = Sigma.unsqueeze(0).repeat((dim,) + (1,)*Sigma.ndim)
            isqrtm_Sigma = Sigma.clone()
        else:
            assert init.shape[-1] == X.shape[-1], \
                    f"Size of initial covariance ({init.shape}) " +\
                    f"incompatible with data ({X.shape})!"
            if init.ndim > 2:
                assert torch.all(batches_dimensions == init.shape[-2]), \
                    f"Size of initial covariance ({init.shape}) " +\
                    f"incompatible with data ({X.shape})!"

            Sigma = init
            isqrtm_Sigma = InvSqrtmEigFunction.apply(Sigma)

        if not self.assume_centered:
            X = X - X.mean(dim=-2, keepdim=True)

        self._init_pbar()
        for _ in range(self.n_iter):
            Sigma_new, isqrtm_Sigma = self._iter_fixed_point(isqrtm_Sigma, X,
                                                             **kwargs)
            delta = torch.norm(Sigma_new - Sigma, "fro") / torch.norm(Sigma, "fro")
            Sigma = Sigma_new
            self._update_pbar(delta)
            if delta < self.tol:
                break
        else:
            if self.verbose:
                warnings.warn("M-estimation didn't converge.")

        if self.verbose:
            self.pbar.close()

        if self.normalize is not None:
            Sigma_new = self.normalize(Sigma_new)

        # For numerical stability
        return .5*(Sigma_new + Sigma_new.transpose(-2,-1))
