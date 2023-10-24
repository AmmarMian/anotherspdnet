# ========================================
# FileName: nn.py
# Date: 11 october 2023 - 14:04
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# Brief: Implementation of torch layers
# =========================================

import os
from typing import Optional

from math import prod
import torch
from torch import nn

from geoopt.manifolds import Stiefel, Sphere
from geoopt.tensor import ManifoldParameter

from .functions import (
        BiMapFunction, ReEigFunction, LogEigFunction, 
        vec_batch, unvec_batch, vech_batch, unvech_batch,
        eig_operation
)
from .utils import initialize_weights_sphere, initialize_weights_stiefel


# =============================================================================
# BiMap layer
# =============================================================================
class BiMap(nn.Module):
    """ BiMap layer in a SPDnet layer according to the paper:
        A Riemannian Network for SPD Matrix Learning, Huang et al
        AAAI Conference on Artificial Intelligence, 2017


        Attributes
        ----------
        n_in : int
            Number of input features.

        n_out : int
            Number of output features.

        n_batches : tuple
            Number of Batches of SPD matrices. It must be a tuple 
            containing at least one batch dimension. Default is (1,).

        device : torch.device
            Device on which the layer is initialized. Default is 'cpu'.

        dtype : torch.dtype
            Data type of the layer. Default is torch.float64.

        W : geoopt.tensor.ManifoldParameter
            Weight matrix of the layer. It is a tensor on either the Stiefel
            or the Sphere manifold.
    """

    def __init__(self, n_in: int, n_out: int, n_batches: Optional[tuple] = None,
                manifold: str = 'stiefel',
                initilization_seed: Optional[int] = None,
                dtype: torch.dtype = torch.float32,
                device: torch.device = torch.device('cpu')) -> None:
        """ Constructor of the BiMap layer

        Parameters
        ----------

        n_in : int
            Number of input features.

        n_out : int
            Number of output features.

        n_batches : tuple
            Number of Batches of SPD matrices. It must be a tuple
            containing at least one batch dimension. Default is None.

        manifold : str, optional
            Manifold on which the layer is initialized. Default is 'stiefel'.
            choice between 'stiefel' and 'sphere'.

        initilization_seed : int, optional
            Seed for the initialization of the weight matrix. Default is None.

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64.

        device : torch.device, optional
            Device on which the layer is initialized. Default is 'cpu'.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_batches = n_batches
        self.device = device
        self.initilization_seed = initilization_seed

        if not manifold in ['stiefel', 'sphere']:
            raise ValueError('manifold must be either stiefel or sphere')

        if not isinstance(device, torch.device):
            raise TypeError('device must be a torch.device')
        
        if manifold == 'stiefel':
            self.manifold = Stiefel()
            initialize_weights = initialize_weights_stiefel
        else:
            self.manifold = Sphere()
            initialize_weights = initialize_weights_sphere

        # Initialize the weight matrix using geoopt
        if n_out > n_in:
            if n_batches is None:
                shape = (n_out, n_in)
            else:
                shape = n_batches + (n_out, n_in)
        else:
            if n_batches is None:
                shape = (n_in, n_out)
            else:
                shape = n_batches + (n_in, n_out)
        self.W = ManifoldParameter(torch.empty(shape, dtype=dtype), manifold=self.manifold)
        self.W = initialize_weights(self.W, seed=initilization_seed)
        self.W = self.W.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the BiMap layer

        Parameters
        ----------
        X : torch.Tensor of shape self.n_batches + (n_matrices, n_in, n_in)
            Batches of input SPD matrices.

        Returns
        -------
        Y : torch.Tensor of shape self.n_batches + (n_matrices, n_out, n_out)
            The output matrices is close to SPD. They need regularization with
            the ReEig layer especially if n_out > n_in.
        """
        if self.n_out < self.n_in:
            _W = self.W.transpose(-2, -1)
        else:
            _W = self.W
        return BiMapFunction.apply(X, _W)

    def __repr__(self) -> str:
        """ Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f'BiMap(n_in={self.n_in}, n_out={self.n_out})'

    def __str__(self) -> str:
        """ String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


# =============================================================================
# ReEig layer
# =============================================================================
class ReEig(nn.Module):
    """ ReEig layer in a SPDnet layer according to the paper:
        A Riemannian Network for SPD Matrix Learning, Huang et al
        AAAI Conference on Artificial Intelligence, 2017

        Attributes
        ----------
        eps : float
            Value of rectification of the eigenvalues. Default is 1e-4.
    """

    def __init__(self, eps: float = 1e-4, use_autograd: bool = False) -> None:
        """ Constructor of the ReEig layer

        Parameters
        ----------
        eps : float, optional
            Value of rectification of the eigenvalues. Default is 1e-4.

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.eps = eps
        self.use_autograd = use_autograd

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ReEig layer

        Parameters
        ----------
        X : torch.Tensor of shape (..., n_features, n_features)
            Batches of input almost-SPD matrices.

        Returns
        -------
        Y : torch.Tensor of shape (..., n_features, n_features)
            The regularized SPD matrices.
        """
        if self.use_autograd:
            operation = lambda X: torch.nn.functional.threshold(
                    X, self.eps, self.eps)
            _, _, res = eig_operation(X, operation)
            return res
        return ReEigFunction.apply(X, self.eps)

    def __repr__(self) -> str:
        """ Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f'ReEig(eps={self.eps}, use_autograd={self.use_autograd})'

    def __str__(self) -> str:
        """ String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


# =============================================================================
# LogEig layer
# =============================================================================
class LogEig(nn.Module):
    """ LogEig layer in a SPDnet layer according to the paper:
        A Riemannian Network for SPD Matrix Learning, Huang et al
        AAAI Conference on Artificial Intelligence, 2017

        Attributes
        ----------

    """
    def __init__(self, use_autograd: bool = False) -> None:
        """ Constructor of the LogEig layer

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.use_autograd = use_autograd

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ReEig layer

        Parameters
        ----------
        X : torch.Tensor of shape (..., n_features, n_features)
            Batches of input almost-SPD matrices.

        Returns
        -------
        Y : torch.Tensor of shape (..., n_features, n_features)
            The regularized SPD matrices.
        """
        if self.use_autograd:
            operation = lambda X: torch.log(X)
            _, _, res = eig_operation(X, operation)
            return res
        return LogEigFunction.apply(X)

    def __repr__(self) -> str:
        """ Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return f'LogEig(auto_grad={self.use_autograd})'

    def __str__(self) -> str:
        """ String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()
        

# =============================================================================
# Vectorization layer
# =============================================================================
class Vectorization(nn.Module):
    """Vectorization of a batch of matrices according to the 
    last two dimensions"""

    def __init__(self) -> None:
        """Vectorization of a batch of matrices according to the 
        last two dimensions"""
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Vectorization layer

        Parameters
        ----------
        X: torch.Tensor of shape (..., n, k)
            Batch of matrices.

        Returns
        -------
        X_vec: torch.Tensor of shape (..., n*k)
            Batch of vectorized matrices.
        """
        return vec_batch(X)

    def inverse_transform(self, X: torch.Tensor, n_rows: int) -> torch.Tensor:
        """ Inverse transform of the Vectorization layer

        Parameters
        ----------
        X: torch.Tensor of shape (..., n_rows*k)
            Batch of vectorized matrices.

        n_rows: int
            Number of rows of the original matrices.

        Returns
        -------
        X_vec: torch.Tensor of shape (..., n_rows, k)
            Batch of matrices.
        """
        return unvec_batch(X, n_rows)

    def __repr__(self) -> str:
        """ Representation of the layer

        Returns
        --------
        str
            Representation of the layer
        """
        return 'Vectorization()'

    def __str__(self) -> str:
        """ String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class Vech(nn.Module):
    """Vech operator of a batch of matrices according to the 
    last two dimensions"""
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Vech layer

        Parameters
        ----------
        X: torch.Tensor of shape (..., n, k)
            Batch of matrices.

        Returns
        -------
        X_vech: torch.Tensor of shape (..., n*(n+1)//2)
            Batch of vech matrices.
        """
        return vech_batch(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Inverse transform of the Vech layer

        Parameters
        ----------
        X: torch.Tensor of shape (..., n*(n+1)//2)
            Batch of vech matrices.

        Returns
        -------
        X_vech: torch.Tensor of shape (..., n, k)
            Batch of matrices.
        """
        return unvech_batch(X)

    def __repr__(self) -> str:
        """ Representation of the layer

        Returns
        --------
        str
            Representation of the layer
        """
        return f'Vech()'

    def __str__(self) -> str:
        """ String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()
