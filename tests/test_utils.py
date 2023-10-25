# Test of anotherspdnet.utils module


from anotherspdnet import utils
from unittest import main, TestCase

from math import prod
import torch
from torch.testing import assert_close
import os

from geoopt.manifolds import Stiefel, SymmetricPositiveDefinite, Sphere
from geoopt import ManifoldParameter

seed = 55555
torch.manual_seed(seed)


class TestInitilizeWeights(TestCase):
    """Testing strategies for initializing weight matrices"""

    def test_initilize_weights_stiefel_increase(self):
        """Test the initialize_weights_stiefel function. Version increase."""
        n_in = 15
        n_out = 40
        n_matrices = 13
        manifold = Stiefel()
        W = ManifoldParameter(torch.empty(n_matrices, n_out, n_in),
                            manifold=manifold)
        W = utils.initialize_weights_stiefel(W, seed)
        assert W.shape == (n_matrices, n_out, n_in)
        test = torch.einsum('...ij,...jk->...ik', W.transpose(-1, -2), W)

        assert_close(test, torch.eye(n_in, dtype=W.dtype, 
                    device=W.device).unsqueeze(0).repeat(n_matrices, 1, 1))


    def test_initilize_weights_stiefel_decrease(self):
        """Test the initialize_weights_stiefel function. Version decrease."""
        n_in = 17
        n_out = 7
        n_matrices = 13
        manifold = Stiefel()
        W = ManifoldParameter(torch.empty(n_matrices, n_in, n_out),
                            manifold=manifold)
        W = utils.initialize_weights_stiefel(W, seed)
        assert W.shape == (n_matrices, n_in, n_out)
        assert_close(torch.einsum('...ij,...jk->...ik', W.transpose(-1, -2), W),
                    torch.eye(n_out, dtype=W.dtype,
                              device=W.device).repeat(n_matrices, 1, 1))


    def test_initilize_weights_sphere_increase(self):
        """Test the initialize_weights_sphere function. Version increase."""
        n_in = 17
        n_out = 50
        n_matrices = 13
        manifold = Sphere()
        W = ManifoldParameter(torch.empty(n_matrices, n_out, n_in),
                            manifold=manifold)
        W = utils.initialize_weights_sphere(W, seed)
        assert W.shape == (n_matrices, n_out, n_in)
        assert_close(torch.linalg.norm(W, dim=-1),
                    torch.ones((n_matrices, n_out), dtype=W.dtype,
                              device=W.device))


    def test_initilize_weights_sphere_decrease(self):
        """Test the initialize_weights_sphere function. Version decrease."""
        n_in = 17
        n_out = 7
        n_matrices = 13
        manifold = Sphere()
        W = ManifoldParameter(torch.empty(n_matrices, n_in, n_out),
                            manifold=manifold)
        W = utils.initialize_weights_sphere(W, seed)
        assert W.shape == (n_matrices, n_in, n_out)
        assert_close(torch.linalg.norm(W, dim=-1),
                    torch.ones((n_matrices, n_in), dtype=W.dtype,
                              device=W.device))


class TestTensorUtils(TestCase):
    """Test of utilities functions on tensors."""

    def test_symmetrize(self):
        """Test the symmetrize function."""
        n_features = 17
        n_matrices = (3, 2, 40)
        X = torch.randn(n_matrices + (n_features, n_features))
        sym_X = utils.symmetrize(X)
        assert sym_X.shape == X.shape
        assert_close(sym_X, sym_X.transpose(-1, -2))
        assert sym_X.dtype == X.dtype


    def test_construct_eigdiff_matrix(self):
        """Test the construct_eigdiff_matrix function."""
        n_features = 17
        n_matrices = (3, 2, 40)
        X = SymmetricPositiveDefinite().random(
                n_matrices + (n_features, n_features))
        eigvals = torch.linalg.eigvalsh(X)
        eigdiff_matrix = utils.construct_eigdiff_matrix(eigvals)
        
        assert eigdiff_matrix.shape == X.shape
        assert_close(eigdiff_matrix.diagonal(dim1=-2, dim2=-1),
                     torch.zeros_like(eigdiff_matrix.diagonal(dim1=-2, dim2=-1)))
        assert eigdiff_matrix.dtype == X.dtype
        assert torch.allclose(eigdiff_matrix, -eigdiff_matrix.transpose(-1, -2))


    def test_zero_offdiag(self):
        X = torch.randn(3, 2, 40, 17, 17)
        X_zero = utils.zero_offdiag(X)
        assert X_zero.shape == X.shape
        assert_close(X_zero.diagonal(dim1=-2, dim2=-1), X.diagonal(dim1=-2, dim2=-1))
        assert X_zero.dtype == X.dtype
        assert_close(X_zero, X_zero.transpose(-1, -2))


if __name__ == "__main__":
    main()


