# Test of another_spdnet.functions

from anotherspdnet import functions
from unittest import TestCase, main

import torch
from torch.testing import assert_close
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.stiefel import Stiefel
from geomstats.geometry.spd_matrices import SPDMatrices

seed = 7777
torch.manual_seed(seed)

# =====================================================
# BiMap
# =====================================================
class TestBiMapPythonFunction(TestCase):
    """Test of the BiMap python function."""

    def setUp(self):
        self.n_in_decrease = 17
        self.n_out_decrease = 7
        self.n_in_increase = 7
        self.n_out_increase = 17
        self.n_matrices = 20
        self.n_batches = 13

    def test_basic2D_decrease(self):
        """Test basic operation on 2D matrices.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        X = SPDMatrices(n_in).random_point()
        W = Stiefel(n_in, n_out).random_point().T

        Y = functions.biMap(X, W)

        assert_close(Y, Y.T)
        assert torch.linalg.det(Y) > 0

    def test_batch_decrease(self):
        """Test with a single dimension of batch.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        n_matrices = self.n_matrices
        X = SPDMatrices(n_in).random_point(n_samples=n_matrices)
        W = Stiefel(n_in, n_out).random_uniform(
                n_samples=1).transpose(-2, -1)

        Y = functions.biMap(X, W)

        assert_close(Y, Y.transpose(-2, -1))
        assert torch.all(torch.linalg.det(Y) > 0)

    def test_basic2D_increase(self):
        """Test basic operation on 2D matrices.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        X = SPDMatrices(n_in).random_point()
        W = Stiefel(n_out, n_in).random_point()
        Y = functions.biMap(X, W)

        assert_close(Y, Y.T)

    def test_batch_increase(self):
        """Test with a single dimension of batch.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        n_matrices = self.n_matrices
        X = SPDMatrices(n_in).random_point(n_samples=n_matrices)
        W = Stiefel(n_out, n_in).random_uniform(
                n_samples=1)
        Y = functions.biMap(X, W)
        assert_close(Y, Y.transpose(-2, -1))

    def test_manybatches_decrease(self):
        """Test with many dimensions of batches.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        n_matrices = self.n_matrices
        n_batches = self.n_batches
        n_batches2 = 5
        n_batches3 = 3

        X = SPDMatrices(n_in).random_point(
                n_samples=n_matrices*n_batches*n_batches2*n_batches3)
        X = X.reshape(n_batches, n_batches2, n_batches3,
                      n_matrices, n_in, n_in)

        # To verify the reshape didn't destruct the SPD property
        assert_close(X, X.transpose(-1, -2))

        W = Stiefel(n_in, n_out).random_uniform(
                n_samples=n_batches*n_batches2*n_batches3)
        W = W.reshape(n_batches, n_batches2,
                      n_batches3, n_in, n_out).transpose(-2, -1)

        Y = functions.biMap(X, W)

        assert_close(Y, Y.transpose(-2, -1))
        assert torch.all(torch.linalg.det(Y) > 0)

    def test_manybatches_increase(self):
        """Test with many dimensions of batches.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        n_matrices = self.n_matrices
        n_batches = self.n_batches
        n_batches2 = 5
        n_batches3 = 3
        X = SPDMatrices(n_in).random_point(
                n_samples=n_matrices*n_batches*n_batches2*n_batches3)
        X = X.reshape(n_batches, n_batches2,
                      n_batches3, n_matrices, n_in, n_in)
        assert_close(X, X.transpose(-1, -2))

        W = Stiefel(n_out, n_in).random_uniform(
            n_samples=n_batches*n_batches2*n_batches3)
        W = W.reshape(n_batches, n_batches2, n_batches3, n_out, n_in)

        Y = functions.biMap(X, W)

        assert_close(Y, Y.transpose(-2, -1))


class TestBiMapGradient(TestCase):
    """Test of the BiMap gradient."""
    def setUp(self):
        self.n_in_decrease = 17
        self.n_out_decrease = 7
        self.n_in_increase = 7
        self.n_out_increase = 17
        self.n_matrices = 20
        self.n_batches = 13

    def test_basic2D_decrease(self):
        """Test basic operation on 2D matrices.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        X = SPDMatrices(n_in).random_point()
        W = Stiefel(n_in, n_out).random_point().T
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(n_out)

        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X)
        assert_close(W.grad, grad_W)

    def test_manybatches_decrease(self):
        """Test with many dimensions of batches.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        n_matrices = self.n_matrices
        n_batches = self.n_batches
        n_batches2 = 5
        n_batches3 = 3

        X = SPDMatrices(n_in).random_point(
                n_samples=n_matrices*n_batches*n_batches2*n_batches3)
        X = X.reshape(n_batches, n_batches2, n_batches3,
                      n_matrices, n_in, n_in)

        W = Stiefel(n_in, n_out).random_uniform(
                n_samples=n_batches*n_batches2*n_batches3)
        W = W.reshape(n_batches, n_batches2,
                      n_batches3, n_in, n_out).transpose(-2, -1)
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(n_out).reshape(1, 1, 1, 1, n_out, n_out).repeat(
                n_batches, n_batches2, n_batches3, n_matrices, 1, 1)

        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X)
        assert_close(W.grad, grad_W)

    def test_basic2D_increase(self):
        """Test basic operation on 2D matrices.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        X = SPDMatrices(n_in).random_point()
        W = Stiefel(n_out, n_in).random_point()
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(n_out)
        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X)
        assert_close(W.grad, grad_W)

    def test_manybatches_increase(self):
        """Test with many dimensions of batches.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        n_matrices = self.n_matrices
        n_batches = self.n_batches
        n_batches2 = 5
        n_batches3 = 3
        X = SPDMatrices(n_in).random_point(
            n_samples=n_matrices*n_batches*n_batches2*n_batches3)
        X = X.reshape(n_batches, n_batches2, n_batches3,
                      n_matrices, n_in, n_in)
        W = Stiefel(n_out, n_in).random_uniform(
            n_samples=n_batches*n_batches2*n_batches3)
        W = W.reshape(n_batches, n_batches2, n_batches3, n_out, n_in)
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(n_out).reshape(1, 1, 1, 1, n_out, n_out).repeat(
            n_batches, n_batches2, n_batches3, n_matrices, 1, 1)
        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X)
        assert_close(W.grad, grad_W)


class TestBiMapTorchFunction(TestCase):
    """Test of the BiMap torch function."""

    def setUp(self):
        self.n_in_decrease = 17
        self.n_out_decrease = 7
        self.n_in_increase = 7
        self.n_out_increase = 17
        self.n_matrices = 20
        self.n_batches = 13

    def test_basic2D_decrease(self):
        """Test basic operation on 2D matrices.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        X = SPDMatrices(n_in).random_point()
        W = Stiefel(n_in, n_out).random_point().T
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.T)
        assert torch.linalg.det(Y) > 0

    def test_batch_decrease(self):
        """Test with a single dimension of batch.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        n_matrices = self.n_matrices
        X = SPDMatrices(n_in).random_point(n_samples=n_matrices)
        W = Stiefel(n_in, n_out).random_uniform(
            n_samples=1).transpose(-2, -1)
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.transpose(-2, -1))
        assert torch.all(torch.linalg.det(Y) > 0)

    def test_basic2D_increase(self):
        """Test basic operation on 2D matrices.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        X = SPDMatrices(n_in).random_point()
        W = Stiefel(n_out, n_in).random_point()
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.T)

    def test_batch_increase(self):
        """Test with a single dimension of batch.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        n_matrices = self.n_matrices
        X = SPDMatrices(n_in).random_point(n_samples=n_matrices)
        W = Stiefel(n_out, n_in).random_uniform(
            n_samples=1)
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.transpose(-2, -1))

    def test_manybatches_decrease(self):
        """Test with many dimensions of batches.
        Version decreasing dimension."""
        n_in = self.n_in_decrease
        n_out = self.n_out_decrease
        n_matrices = self.n_matrices
        n_batches = self.n_batches
        n_batches2 = 5
        n_batches3 = 3
        X = SPDMatrices(n_in).random_point(
            n_samples=n_matrices*n_batches*n_batches2*n_batches3)
        X = X.reshape(n_batches, n_batches2,
                      n_batches3, n_matrices, n_in, n_in)
        assert_close(X, X.transpose(-1, -2))
        W = Stiefel(n_in, n_out).random_uniform(
            n_samples=n_batches*n_batches2*n_batches3)
        W = W.reshape(
                n_batches, n_batches2,
                n_batches3, n_in, n_out).transpose(-2, -1)
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.transpose(-2, -1))
        assert torch.all(torch.linalg.det(Y) > 0)

    def test_manybatches_increase(self):
        """Test with many dimensions of batches.
        Version increasing dimension."""
        n_in = self.n_in_increase
        n_out = self.n_out_increase
        n_matrices = self.n_matrices
        n_batches = self.n_batches
        n_batches2 = 5
        n_batches3 = 3
        X = SPDMatrices(n_in).random_point(
            n_samples=n_matrices*n_batches*n_batches2*n_batches3)
        X = X.reshape(n_batches, n_batches2, n_batches3,
                      n_matrices, n_in, n_in)
        assert_close(X, X.transpose(-1, -2))
        W = Stiefel(n_out, n_in).random_uniform(
            n_samples=n_batches*n_batches2*n_batches3)
        W = W.reshape(n_batches, n_batches2, n_batches3, n_out, n_in)
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.transpose(-2, -1))


if __name__ == '__main__':
    main()
