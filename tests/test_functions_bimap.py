# Test of another_spdnet.functions relative to bimap

from anotherspdnet import functions
from unittest import TestCase, main

import torch
from torch.testing import assert_close
import os

from geoopt.manifolds import Stiefel, SymmetricPositiveDefinite

seed = 7777
torch.manual_seed(seed)

# =====================================================
# BiMap
# =====================================================
class TestBiMapPythonFunction(TestCase):
    """Test of the BiMap python function."""

    def setUp(self):
        self.n_in = 17
        self.n_out_decrease = 7
        self.n_out_increase = 50
        self.n_matrices = 20
        self.n_batches = 13
        self.X = SymmetricPositiveDefinite().random(
                (self.n_batches, self.n_matrices, self.n_in, self.n_in))
        self.W_decrease = Stiefel().random(
                (self.n_batches, self.n_in,
                 self.n_out_decrease)).transpose(-2, -1)
        self.W_increase = Stiefel().random(
            (self.n_batches, self.n_out_increase, self.n_in))


    def test_basic2D_decrease(self):
        """Test basic operation on 2D matrices.
        Version decreasing dimension."""
        X = self.X[0, 0]
        W = self.W_decrease[0]

        Y = functions.biMap(X, W)

        assert_close(Y, Y.T)
        assert torch.linalg.det(Y) > 0

    def test_batch_decrease(self):
        """Test with a single dimension of batch.
        Version decreasing dimension."""

        Y = functions.biMap(self.X, self.W_decrease)

        assert_close(Y, Y.transpose(-2, -1))
        assert torch.all(torch.linalg.det(Y) > 0)

    def test_basic2D_increase(self):
        """Test basic operation on 2D matrices.
        Version increasing dimension."""
        X = self.X[0, 0]
        W = self.W_increase[0]
        Y = functions.biMap(X, W)

        assert_close(Y, Y.T)

    def test_batch_increase(self):
        """Test with a single dimension of batch.
        Version increasing dimension."""
        Y = functions.biMap(self.X, self.W_increase)
        assert_close(Y, Y.transpose(-2, -1))


class TestBiMapGradient(TestCase):
    """Test of the BiMap gradient."""
    def setUp(self):
        self.n_in = 17
        self.n_out_decrease = 7
        self.n_out_increase = 50
        self.n_matrices = 27
        self.n_batches1 = 13
        self.n_batches2 = 20
        self.X = SymmetricPositiveDefinite().random(
                (self.n_batches1, self.n_batches2, self.n_matrices,
                 self.n_in, self.n_in))
        self.W_decrease = Stiefel().random(
                (self.n_batches1, self.n_batches2, self.n_in,
                self.n_out_decrease)).transpose(-2, -1)
        self.W_increase = Stiefel().random(
                (self.n_batches1, self.n_batches2, self.n_out_increase,
                 self.n_in))

    def test_basic2D_decrease(self):
        """Test basic operation on 2D matrices.
        Version decreasing dimension."""
        X = self.X[0, 0, 0]
        W = self.W_decrease[0, 0]
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(self.n_out_decrease)

        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X, atol=1e-4, rtol=1e-4)
        assert_close(W.grad, grad_W, atol=1e-4, rtol=1e-4)

    def test_manybatches_decrease(self):
        """Test with many dimensions of batches.
        Version decreasing dimension."""
        X = self.X
        W = self.W_decrease
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(self.n_out_decrease).reshape(
                1, 1, 1, self.n_out_decrease, self.n_out_decrease).repeat(
                self.n_batches1, self.n_batches2, self.n_matrices, 1, 1)

        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X, atol=1e-4, rtol=1e-4)
        assert_close(W.grad, grad_W, atol=1e-4, rtol=1e-4)

    def test_basic2D_increase(self):
        """Test basic operation on 2D matrices.
        Version increasing dimension."""
        X = self.X[0, 0, 0]
        W = self.W_increase[0, 0]
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(self.n_out_increase)
        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X, atol=1e-4, rtol=1e-4)
        assert_close(W.grad, grad_W, atol=1e-4, rtol=1e-4)

    def test_manybatches_increase(self):
        """Test with many dimensions of batches.
        Version increasing dimension."""
        X = self.X
        W = self.W_increase
        X.requires_grad = True
        W.requires_grad = True
        Y = functions.biMap(X, W)

        # Mockup loss = Trace
        loss = torch.einsum('...ii->', Y)
        grad_Y = torch.eye(self.n_out_increase).reshape(
            1, 1, 1, self.n_out_increase, self.n_out_increase).repeat(
            self.n_batches1, self.n_batches2, self.n_matrices, 1, 1)
        grad_X, grad_W = functions.biMap_gradient(X, W, grad_Y)
        assert grad_X.shape == X.shape
        assert grad_W.shape == W.shape

        # Comparing to autograd
        loss.backward()
        assert_close(X.grad, grad_X, atol=1e-4, rtol=1e-4)
        assert_close(W.grad, grad_W, atol=1e-4, rtol=1e-4)


class TestBiMapTorchFunction(TestCase):
    """Test of the BiMap torch function."""

    def setUp(self):
        self.n_in = 40
        self.n_out_decrease = 7
        self.n_out_increase = 70
        self.n_matrices = 20
        self.n_batches = 13
        self.X = SymmetricPositiveDefinite().random(
                (self.n_batches, self.n_matrices, self.n_in, self.n_in))
        self.W_decrease = Stiefel().random(
                (self.n_batches, self.n_in,
                self.n_out_decrease)).transpose(-2, -1)
        self.W_increase = Stiefel().random(
                (self.n_batches, self.n_out_increase,
                 self.n_in))


    def test_basic2D_decrease(self):
        """Test basic operation on 2D matrices.
        Version decreasing dimension."""
        X = self.X[0, 0]
        W = self.W_decrease[0]
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.T)
        assert torch.linalg.det(Y) > 0

    def test_batch_decrease(self):
        """Test with a single dimension of batch.
        Version decreasing dimension."""
        X = self.X
        W = self.W_decrease
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.transpose(-2, -1))
        assert torch.all(torch.linalg.det(Y) > 0)

    def test_basic2D_increase(self):
        """Test basic operation on 2D matrices.
        Version increasing dimension."""
        X = self.X[0, 0]
        W = self.W_increase[0]
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.T)

    def test_batch_increase(self):
        """Test with a single dimension of batch.
        Version increasing dimension."""
        X = self.X
        W = self.W_increase
        Y = functions.BiMapFunction.apply(X, W)
        assert_close(Y, Y.transpose(-2, -1))


if __name__ == '__main__':
    main()
