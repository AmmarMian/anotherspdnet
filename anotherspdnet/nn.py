# ========================================
# FileName: nn.py
# Date: 11 october 2023 - 14:04
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# Brief: Implementation of torch layers
# =========================================

import os

from math import prod
import torch
from torch import nn

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.stiefel import Stiefel

from .parameters import StiefelParameter
from .functions import (
        BiMapFunction, ReEigFunction, LogEigFunction, 
        vec_batch, unvec_batch, vech_batch, unvech_batch
)

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

        W : StiefelParameter
            Weight matrix of the layer. It is initialized randomly
            on the Stiefel manifold.
    """

    def __init__(self, n_in: int, n_out: int, n_batches: tuple = (1,),
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
            containing at least one batch dimension. Default is (1,).

        device : torch.device, optional
            Device on which the layer is initialized. Default is 'cpu'.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_batches = n_batches
        n_matrices = prod(n_batches)

        # Initialize the weight matrix using geomstats
        if n_out > n_in:
            self.stiefel = Stiefel(n_out, n_in)
            W = self.stiefel.random_uniform(n_matrices)
            W = W.reshape(n_batches + (n_out, n_in))
        else:
            self.stiefel = Stiefel(n_in, n_out)
            W = self.stiefel.random_uniform(n_matrices)
            W = W.reshape(n_batches + (n_in, n_out))

        W = W.to(device)
        self.W = StiefelParameter(W, requires_grad=True)

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
        """ Representation of the layer """
        return f'BiMap(n_in={self.n_in}, n_out={self.n_out})'

    def __str__(self) -> str:
        """ String representation of the layer """
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

    def __init__(self, eps: float = 1e-4) -> None:
        """ Constructor of the ReEig layer

        Parameters
        ----------
        eps : float, optional
            Value of rectification of the eigenvalues. Default is 1e-4.
        """
        super().__init__()
        self.eps = eps

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
        return ReEigFunction.apply(X, self.eps)

    def __repr__(self) -> str:
        """ Representation of the layer """
        return f'ReEig(eps={self.eps})'

    def __str__(self) -> str:
        """ String representation of the layer """
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
    def __init__(self) -> None:
        """ Constructor of the LogEig layer"""
        super().__init__()

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
        return LogEigFunction.apply(X)

    def __repr__(self) -> str:
        """ Representation of the layer """
        return f'LogEig()'

    def __str__(self) -> str:
        """ String representation of the layer """
        return self.__repr__()
        

# =============================================================================
# Vectorization layer
# =============================================================================
class Vectorization(nn.Module):
    """Vectorization of a batch of matrices according to the 
    last two dimensions"""

    def __init__(self, n_rows: int) -> None:
        """ Constructor of the Vectorization layer
        Parameters
        ----------
        n_rows : int
            Number of rows of the matrices.
        """
        super().__init__()
        self.n_rows = n_rows

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

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """ Inverse transform of the Vectorization layer

        Parameters
        ----------
        X: torch.Tensor of shape (..., n*k)
            Batch of vectorized matrices.
        Returns
        -------
        X_vec: torch.Tensor of shape (..., n, k)
            Batch of matrices.
        """
        return unvec_batch(X, self.n_rows)

    def __repr__(self) -> str:
        """ Representation of the layer """
        return f'Vectorization(n_rows={self.n_rows})'

    def __str__(self) -> str:
        """ String representation of the layer """
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
        """ Representation of the layer """
        return f'Vech()'

    def __str__(self) -> str:
        """ String representation of the layer """
        return self.__repr__()
