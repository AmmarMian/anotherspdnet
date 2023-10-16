# ========================================
# FileName: nn.py
# Date: 11 october 2023 - 14:04
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# Brief: Implementation of torch layers
# =========================================

import os

import torch
from torch import nn

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.stiefel import Stiefel

from .parameters import StiefelParameter
from .functions import BiMapFunction

# =============================================================================
# BiMap layer
# =============================================================================
class BiMap(nn.Module):
    """ BiMap layer in a SPDnet layer according to the paper:
        A Riemannian Network for SPD Matrix Learning, Huang et al
        AAAI Conference on Artificial Intelligence, 2017
    """

    def __init__(self, n_batches: int, n_in: int, n_out: int,
                device: torch.device = torch.device('cpu')) -> None:
        """ Constructor of the BiMap layer

        Parameters
        ----------
        n_batches : int
            Number of Batches.

        n_in : int
            Number of input features.

        n_out : int
            Number of output features.

        device : torch.device, optional
            Device on which the layer is initialized. Default is 'cpu'.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        # Initialize the weight matrix using geomstats
        if n_out > n_in:
            self.stiefel = Stiefel(n_out, n_in)
            W = self.stiefel.random_uniform(n_batches)
        else:
            self.stiefel = Stiefel(n_in, n_out)
            W = self.stiefel.random_uniform(n_batches)
            W = W.transpose(-2, -1)

        if n_batches == 1:
            W = W.unsqueeze(0)
        print(W.shape)

        W = W.to(device)
        self.W = StiefelParameter(W, requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the BiMap layer

        Parameters
        ----------
        X : torch.Tensor of shape (n_batches, n_matrices, n_in, n_in)
            Batches of input SPD matrices.

        Returns
        -------
        Y : torch.Tensor of shape (n_batches, n_matrices, n_out, n_out)
            The output matrices. SPD if n_out <= n_in. Otherwise, they
            need regularization.
        """
        return BiMapFunction.apply(X, self.W)

    def extra_repr(self) -> str:
        """ Extra representation of the layer """
        return f'n_in={self.n_in}, n_out={self.n_out}'

    def __repr__(self) -> str:
        """ Representation of the layer """
        return f'BiMap({self.n_in}, {self.n_out})'

    def __str__(self) -> str:
        """ String representation of the layer """
        return self.__repr__()
