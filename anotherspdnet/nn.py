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
from .functions import BiMapFunction, ReEigFunction, LogEigFunction

# =============================================================================
# BiMap layer
# =============================================================================
class BiMap(nn.Module):
    """ BiMap layer in a SPDnet layer according to the paper:
        A Riemannian Network for SPD Matrix Learning, Huang et al
        AAAI Conference on Artificial Intelligence, 2017
    """

    def __init__(self,n_in: int, n_out: int, n_batches: tuple = (1,),
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
        else:
            self.stiefel = Stiefel(n_in, n_out)
            W = self.stiefel.random_uniform(n_matrices)
            W = W.transpose(-2, -1)
        W = W.reshape(n_batches + (n_out, n_in))

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
        return BiMapFunction.apply(X, self.W)

    def __repr__(self) -> str:
        """ Representation of the layer """
        return f'BiMap({self.n_in}, {self.n_out})'

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


# =============================================================================
# LogEig layer
# =============================================================================
class LogEig(nn.Module):
    """ LogEig layer in a SPDnet layer according to the paper:
        A Riemannian Network for SPD Matrix Learning, Huang et al
        AAAI Conference on Artificial Intelligence, 2017
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
        

