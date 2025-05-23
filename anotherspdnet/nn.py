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
    BiMapFunction,
    ReEigFunction,
    LogEigFunction,
    vec_batch,
    unvec_batch,
    vech_batch,
    unvech_batch,
    eig_operation,
    biMap,
    ReEigBiasFunction,
)
from .utils import initialize_weights_sphere, initialize_weights_stiefel


# =============================================================================
# BiMap layer
# =============================================================================
class BiMap(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_batches: Optional[tuple] = None,
        manifold: str = "stiefel",
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        mm_mode: str = "einsum",
        use_autograd: bool = False,
    ) -> None:
        """BiMap layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

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

        seed : int, optional
            Seed for the initialization of the weight matrix. Default is None.

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64.

        device : torch.device, optional
            Device on which the layer is initialized. Default is 'cpu'.

        mm_mode : str, optional
            Mode for the matrix multiplication. Default is 'einsum'.
            Choice between 'einsum' and 'bmm'.

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_batches = n_batches
        self.device = device
        self.seed = seed
        self.dtype = dtype
        self.use_autograd = use_autograd
        self.dim = n_out

        if mm_mode not in ["einsum", "bmm"]:
            raise ValueError("mm_mode must be either einsum or bmm")
        self.mm_mode = mm_mode

        if manifold not in ["stiefel", "sphere"]:
            raise ValueError("manifold must be either stiefel or sphere")

        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")

        if manifold == "stiefel":
            self.manifold = Stiefel()
            initialize_weights = initialize_weights_stiefel
        else:
            self.manifold = Sphere()
            initialize_weights = initialize_weights_sphere

        # Initialize the weight matrix using geoopt.
        # We need to keep a Stiefel matrix in both cases because of the
        # Riemanian optimization.
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
        _W = ManifoldParameter(
            torch.empty(shape, dtype=dtype, device=device),
            manifold=self.manifold,
        )
        self.W = initialize_weights(_W, seed=seed).type(self.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BiMap layer

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

        if self.use_autograd:
            return biMap(X, _W, self.mm_mode)

        return BiMapFunction.apply(X, _W, self.mm_mode)

    def __repr__(self) -> str:
        """Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f'BiMap(n_in={self.n_in}, n_out={self.n_out}, " \
                f"shape={self.W.shape}, use_autograd={self.use_autograd}, '
            f"seed={self.seed}, dtype={self.dtype}, device={self.device}, "
            f"mm_mode={self.mm_mode})"
        )

    def __str__(self) -> str:
        """String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


# =============================================================================
# ReEig layer
# =============================================================================
class ReEigBias(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-4,
        use_autograd: bool = False,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> None:
        """ReEig layer with a bias term.

        Parameters
        ----------
        dim : int
            Dimension of the SPD matrices.

        eps : float, optional
            Value of rectification of the eigenvalues. Default is 1e-4.

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is True.
            FOR NOW: without autograd, the layer is not implemented.

        dtype : torch.dtype, optional
            Data type of the layer. Default is torch.float64.

        device : torch.device, optional
            Device on which the layer is initialized. Default is 'cpu'.

        seed : int, optional
            Seed for the initialization of the bias term. Default is None.
        """

        super().__init__()
        self.eps = eps
        self.dim = dim
        self.use_autograd = use_autograd
        self.dtype = dtype
        self.seed = seed

        # Initialize the bias term
        # init_bias = torch.randn(dim, dtype=self.dtype,
        #                         generator=torch.Generator().manual_seed(seed))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=self.dtype, device=device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ReEig layer with bias

        Parameters
        ----------
        X : torch.Tensor of shape (..., n_features, n_features)
            Batches of input almost-SPD matrices.

        Returns
        -------
        Y : torch.Tensor of shape (..., n_features, n_features)
            The regularized SPD matrices.
        """
        assert X.shape[-1] == self.dim, f"Input matrices must have dimension {self.dim}"

        if self.use_autograd:
            operation = lambda x: torch.min(
                torch.nn.functional.threshold(x + self.bias, self.eps, self.eps),
                (1 / self.eps) * torch.ones_like(x),
            )
            _, _, res = eig_operation(X, operation)

        else:
            res = ReEigBiasFunction.apply(X, self.bias, self.eps)

        return res

    def __repr__(self) -> str:
        """Representation of the layer
        Returns
        -------
        str
            Representation of the layer
        """
        return f'ReEigBias(dim={self.dim}, eps={self.eps}, " \
                f"use_autograd={self.use_autograd}, seed={self.seed})'


class ReEig(nn.Module):
    def __init__(
        self,
        eps: float = 1e-4,
        use_autograd: bool = False,
        mm_mode: str = "einsum",
        eig_function: str = "eigh",
        formula: str = "brooks",
        dim: Optional[int] = None,
    ) -> None:
        """ReEig layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

        Parameters
        ----------
        eps : float, optional
            Value of rectification of the eigenvalues. Default is 1e-4.

        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.

        mm_mode : str, optional
            Mode for the matrix multiplication. Default is 'einsum'. Choice
            between 'einsum' and 'bmm'.

        eig_function : str, optional
            Function used for the computation of the eigendecomposition.
            Default is 'eigh'. Choice between 'eigh' and 'eig'.

        formula : str, optional
            Formula used for the computation of the gradient. Default is
                'brooks'. Choice between 'brooks' and 'ionescu".

        dim : int, optional
            Dimension of the SPD matrices. Default is None.
            Used for logging purposes.
        """
        super().__init__()
        self.eps = eps
        self.use_autograd = use_autograd
        self.dim = dim
        self.mm_mode = mm_mode
        self.eig_function = eig_function
        self.formula = formula

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
            operation = lambda X: torch.nn.functional.threshold(X, self.eps, self.eps)
            _, _, res = eig_operation(X, operation, self.eig_function, self.mm_mode)
            return res
        return ReEigFunction.apply(
            X, self.eps, self.mm_mode, self.eig_function, self.formula
        )

    def __repr__(self) -> str:
        """Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        base_str = (
            f"ReEig(eps={self.eps}, use_autograd={self.use_autograd}, "
            f"mm_mode={self.mm_mode}, eig_function={self.eig_function}, "
            f"formula={self.formula}"
        )
        if self.dim is None:
            base_str += ")"
        else:
            base_str += f", dim={self.dim})"
        return base_str

    def __str__(self) -> str:
        """String representation of the layer

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
    def __init__(
        self,
        use_autograd: bool = False,
        mm_mode: str = "einsum",
        eig_function: str = "eigh",
        formula: str = "brooks",
    ) -> None:
        """LogEig layer in a SPDnet layer according to the paper:
            A Riemannian Network for SPD Matrix Learning, Huang et al
            AAAI Conference on Artificial Intelligence, 2017

        Parameters
        ----------
        use_autograd : bool, optional
            Use torch autograd for the computation of the gradient rather than
            the analytical formula. Default is False.

        mm_mode : str, optional
            Mode for the matrix multiplication. Default is 'einsum'. Choice
            between 'einsum' and 'bmm'.

        eig_function : str, optional
            Function used for the computation of the eigendecomposition.
            Default is 'eigh'. Choice between 'eigh' and 'eig'.

        formula : str, optional
            Formula used for the computation of the gradient. Default is
            'brooks'. Choice between 'brooks' and 'ionescu".
        """
        super().__init__()
        self.use_autograd = use_autograd
        self.mm_mode = mm_mode
        self.eig_function = eig_function
        self.formula = formula

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
            _, _, res = eig_operation(X, operation, self.eig_function, self.mm_mode)
            return res
        return LogEigFunction.apply(X, self.mm_mode, self.eig_function, self.formula)

    def __repr__(self) -> str:
        """Representation of the layer

        Returns
        -------
        str
            Representation of the layer
        """
        return (
            f"LogEig(auto_grad={self.use_autograd}, mm_mode={self.mm_mode}, "
            f"eig_function={self.eig_function}, formula={self.formula})"
        )

    def __str__(self) -> str:
        """String representation of the layer

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
        """Inverse transform of the Vectorization layer

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
        """Representation of the layer

        Returns
        --------
        str
            Representation of the layer
        """
        return "Vectorization()"

    def __str__(self) -> str:
        """String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()


class Vech(nn.Module):
    def __init__(self) -> None:
        """Vech operator of a batch of matrices according to the
        last two dimensions"""
        super().__init__()

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
        """Representation of the layer

        Returns
        --------
        str
            Representation of the layer
        """
        return f"Vech()"

    def __str__(self) -> str:
        """String representation of the layer

        Returns
        -------
        str
            String representation of the layer
        """
        return self.__repr__()
