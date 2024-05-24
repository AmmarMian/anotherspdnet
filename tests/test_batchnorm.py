# Test of another_spdnet.batchnorm

from anotherspdnet import batchnorm
from unittest import TestCase, main

import torch
from torch.testing import assert_close

from geoopt.manifolds import SymmetricPositiveDefinite, Sphere

seed = 7777
torch.manual_seed(seed)

# Change default torch to float64
torch.set_default_dtype(torch.float64)


class TestRiemannianMean(TestCase):

    def test_noweights(self):
        """Test computing Riemannian mean without weights"""
        X = SymmetricPositiveDefinite().random((100, 30, 7, 7))
        X_hat = batchnorm.riemannian_mean_spd(X)
        self.assertEqual(X_hat.shape, (7, 7))
        assert_close(X_hat, X_hat.T)


    def test_weights(self):
        """Test computing Riemannian mean with weights"""
        X = SymmetricPositiveDefinite().random((100, 30, 7, 7))
        weights = torch.abs(torch.randn((100, 30)))
        X_hat = batchnorm.riemannian_mean_spd(X, weights=weights)
        self.assertEqual(X_hat.shape, (7, 7))
        assert_close(X_hat, X_hat.T)

class TestBatchNormSPD(TestCase):

    def test_init(self):
        """Test BatchNormSPD layer init"""
        n_features = 49
        layer = batchnorm.BatchNormSPD(n_features=n_features)
        self.assertEqual(layer.n_features, n_features)
        self.assertEqual(layer.running_mean.shape, (n_features, n_features))
        self.assertEqual(layer.running_mean.shape, (n_features, n_features))

    def test_forward(self):
        """Test forward of BatchNormSPD layer"""
        X = SymmetricPositiveDefinite().random((100, 30, 7, 7))
        layer = batchnorm.BatchNormSPD(n_features=7)
        Y = layer.forward(X)
        self.assertEqual(Y.shape, X.shape)
        self.assertEqual(layer.running_mean.shape, (7, 7))

    def test_backward(self):
        """Test backward of BatchNormSPD layer"""
        X = SymmetricPositiveDefinite().random((100, 30, 7, 7))
        X.requires_grad = True
        layer = batchnorm.BatchNormSPD(n_features=7)
        self.assertIsNone(X.grad)
        self.assertIsNone(layer.bias.grad)
        Y = layer.forward(X)
        loss = torch.einsum('...ii->', Y)
        loss.backward()
        self.assertIsNotNone(layer.bias.grad)
        self.assertIsNotNone(X.grad)

    def test_repr(self):
        layer = batchnorm.BatchNormSPD(n_features=7)
        self.assertIsInstance(layer.__repr__(), str)

    def test_repr(self):
        layer = batchnorm.BatchNormSPD(n_features=7)
        self.assertIsInstance(str(layer), str)

if __name__ == "__main__":
    main()
