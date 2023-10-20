# Test of another_spdnet.nn

from anotherspdnet import nn
from unittest import TestCase, main

from math import prod
import torch
from torch.testing import assert_close
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.spd_matrices import SPDMatrices

seed = 7777
torch.manual_seed(seed)


# =============================================================================
# BiMap layer
# =============================================================================
class TestBiMap(TestCase):
    """ Test the BiMap layer """

    def setUp(self) -> None:
        self.n_batches = (3,)
        self.n_batches_many = (3, 2, 5, 7)
        self.n_matrices = 5
        self.n_in_decrease = 17
        self.n_out_decrease = 7
        self.n_in_increase = 7
        self.n_out_increase = 17

    def test_init(self) -> None:
        """ Test the initialization of the BiMap layer """
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease,
                         self.n_batches)
        self.assertEqual(layer.n_in, self.n_in_decrease)
        self.assertEqual(layer.n_out, self.n_out_decrease)
        self.assertEqual(layer.W.shape, (self.n_batches + (self.n_in_decrease,
                                        self.n_out_decrease)))

    def test_forward_decrease(self) -> None:
        """ Test the forward pass of the BiMap layer.
        Version decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease,
                         self.n_batches)
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_in_decrease).random_point(
                n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches + (self.n_matrices, self.n_in_decrease,
                    self.n_in_decrease))

        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches +(self.n_matrices,
                                self.n_out_decrease, self.n_out_decrease)))

    def test_forward_increase(self) -> None:
        """ Test the forward pass of the BiMap layer.
        Version increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase,
                         self.n_batches)
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_in_increase).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches +(self.n_matrices, self.n_in_increase,
                self.n_in_increase))
        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches + (self.n_matrices,
                            self.n_out_increase, self.n_out_increase)))

    def test_backward_decrease(self) -> None:
        """ Test the backward pass of the BiMap layer.
        Vesion decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease,
                         self.n_batches)
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_in_decrease).random_point(
                n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches + (self.n_matrices, self.n_in_decrease,
                    self.n_in_decrease))
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(X.grad.shape, (self.n_batches +(self.n_matrices,
                                self.n_in_decrease, self.n_in_decrease)))

    def test_backward_increase(self) -> None:
        """ Test the backward pass of the BiMap layer.
        Version increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase,
                         self.n_batches)
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_in_increase).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches +(self.n_matrices, self.n_in_increase,
                self.n_in_increase))
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(X.grad.shape, (self.n_batches +(self.n_matrices,
                            self.n_in_increase, self.n_in_increase)))

    def test_init_gpu(self) -> None:
        """Test when cuda is available the initialization of the BiMap layer.
        Do nothing if cuda is not available."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, 
                             self.n_batches, device=device)
            self.assertEqual(layer.n_in, self.n_in_decrease)
            self.assertEqual(layer.n_out, self.n_out_decrease)
            self.assertEqual(layer.W.shape, (self.n_batches +
                                             (self.n_out_decrease,
                                              self.n_in_decrease)))

    def test_forward_decrease_many(self) -> None:
        """ Test the forward pass of the BiMap layer with many batches 
        dimensions. Version decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease,
                         self.n_batches_many)
        n_batches_total = prod(self.n_batches_many)
        X = SPDMatrices(self.n_in_decrease).random_point(
                n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches_many + (self.n_matrices, self.n_in_decrease,
                    self.n_in_decrease))

        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches_many +(self.n_matrices,
                                self.n_out_decrease, self.n_out_decrease)))

    def test_forward_increase_many(self) -> None:
        """ Test the forward pass of the BiMap layer with many batches 
        dimensions. Version increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase,
                         self.n_batches_many)
        n_batches_total = prod(self.n_batches_many)
        X = SPDMatrices(self.n_in_increase).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches_many +(self.n_matrices, self.n_in_increase,
                self.n_in_increase))
        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches_many + (self.n_matrices,
                            self.n_out_increase, self.n_out_increase)))

    def test_backward_decrease_many(self) -> None:
        """ Test the backward pass of the BiMap layer with many batches 
        dimensions. Version decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease,
                         self.n_batches_many)
        n_batches_total = prod(self.n_batches_many)
        X = SPDMatrices(self.n_in_decrease).random_point(
                n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches_many + (self.n_matrices, self.n_in_decrease,
                    self.n_in_decrease))
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(X.grad.shape, (self.n_batches_many +(self.n_matrices,
                                self.n_in_decrease, self.n_in_decrease)))

    def test_backward_increase(self) -> None:
        """ Test the backward pass of the BiMap layer with many batches 
        dimensions. Vesion increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase,
                         self.n_batches_many)
        n_batches_total = prod(self.n_batches_many)
        X = SPDMatrices(self.n_in_increase).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches_many +(self.n_matrices, self.n_in_increase,
                self.n_in_increase))
        X.requires_grad = True
        Y = layer(X)
        loss = torch.einsum('...ii->', Y)
        loss.backward()
        self.assertEqual(X.grad.shape, (self.n_batches_many +(self.n_matrices,
                            self.n_in_increase, self.n_in_increase)))
        assert_close(X.grad, X.grad.transpose(-1, -2))

    def test_deviceisrespected(self) -> None:
        """Test if the device is respected when initializing the BiMap layer"""
        # Cuda if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, 
                         self.n_batches, device=device)
            self.assertEqual(layer.W.device, device)

            n_batches_total = prod(self.n_batches_many)
            X = SPDMatrices(self.n_in_increase).random_point(
                n_samples=self.n_matrices*n_batches_total)
            X = X.reshape(self.n_batches_many +(self.n_matrices,
                            self.n_in_increase, self.n_in_increase))
            Y = layer(X)
            self.assertEqual(Y.device, device)


# =============================================================================
# Test of the ReEig layer
# =============================================================================
class TestReEig(TestCase):
    """ Test the ReEig layer """

    def setUp(self) -> None:
        self.n_batches = (3,)
        self.n_batches_many = (3, 2, 5, 7)
        self.n_matrices = 5
        self.n_features = 7
        self.eps = 1e-4

    def test_init(self) -> None:
        """ Test the initialization of the ReEig layer """
        layer = nn.ReEig(self.eps)
        self.assertEqual(layer.eps, self.eps)

    def test_forward(self) -> None:
        """ Test the forward pass of the ReEig layer """
        layer = nn.ReEig(self.eps)
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_features).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches + (self.n_matrices, self.n_features,
                self.n_features))
        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches + (self.n_matrices,
                            self.n_features, self.n_features)))

    def test_backward(self) -> None:
        """ Test the backward pass of the ReEig layer """
        layer = nn.ReEig(self.eps)
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_features).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches + (self.n_matrices, self.n_features,
            self.n_features))
        X.requires_grad = True
        Y = layer(X)
        loss = torch.einsum('...ii->', Y)
        loss.backward()
        self.assertEqual(X.grad.shape, (self.n_batches + (self.n_matrices,
                        self.n_features, self.n_features)))

    def test_deviceisrespected(self) -> None:
        """Test if the device is respected for ReEig layer"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            layer = nn.ReEig(self.eps)
            n_batches_total = prod(self.n_batches)
            X = SPDMatrices(self.n_features).random_point(
                n_samples=self.n_matrices*n_batches_total)
            X = X.reshape(self.n_batches + (self.n_matrices, self.n_features,
                self.n_features))
            X.requires_grad = True
            Y = layer(X)
            assert Y.device == device


# =============================================================================
# LogEig layer
# =============================================================================
class TestLogEig(TestCase):
    """ Test the LogEig layer """

    def setUp(self) -> None:
        self.n_batches = (3,)
        self.n_batches_many = (3, 2, 5, 7)
        self.n_matrices = 5
        self.n_features = 7
        self.eps = 1e-4

    def test_init(self) -> None:
        """ Test the initialization of the LogEig layer """
        _ = nn.LogEig()

    def test_forward(self) -> None:
        """ Test the forward pass of the ReEig layer """
        layer = nn.LogEig()
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_features).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches + (self.n_matrices, self.n_features,
                self.n_features))
        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches + (self.n_matrices,
                            self.n_features, self.n_features)))

    def test_backward(self) -> None:
        """ Test the backward pass of the LogEig layer """
        layer = nn.LogEig()
        n_batches_total = prod(self.n_batches)
        X = SPDMatrices(self.n_features).random_point(
            n_samples=self.n_matrices*n_batches_total)
        X = X.reshape(self.n_batches + (self.n_matrices, self.n_features,
            self.n_features))
        X.requires_grad = True
        Y = layer(X)
        loss = torch.einsum('...ii->', Y)
        loss.backward()
        self.assertEqual(X.grad.shape, (self.n_batches + (self.n_matrices,
                        self.n_features, self.n_features)))


if __name__ == '__main__':
    main()
