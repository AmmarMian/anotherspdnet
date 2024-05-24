# Test of another_spdnet.estimation

from anotherspdnet import estimation
from unittest import TestCase, main

import torch
from torch.testing import assert_close


seed = 7777
torch.manual_seed(seed)

# Change default torch to float64
torch.set_default_dtype(torch.float64)


class TestMestFunctions(TestCase):

    def test_tyler_function(self):
        """Test Tyler function"""
        X = torch.randn((50, 30, 20, 10))
        Y = estimation.tyler_function(X, 10)
        self.assertEqual(X.shape, Y.shape)

    def test_huber_function(self):
        """Test Huber function"""
        X = torch.randn((50, 30, 20, 10))
        Y = estimation.huber_function(X, 0.5, 1)
        self.assertEqual(X.shape, Y.shape)

    def test_student_function(self):
        """Test Student-t function"""
        X = torch.randn((50, 30, 20, 10))
        Y = estimation.student_function(X, 10, 3)
        self.assertEqual(X.shape, Y.shape)


class TestMestLayer(TestCase):

    def test_init(self):
        estimation.Mestimation(lambda x: 1/x)

    def test_forward_tyler(self):
        X = torch.randn((50, 20, 100, 9))
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.tyler_function 
                )
        Sigma = layer.forward(X, n_features=9)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_forward_student(self):
        X = torch.randn((50, 20, 100, 9))
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.student_function
                )
        Sigma = layer.forward(X, n_features=9, nu=2)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_forward_huber(self):
        X = torch.randn((50, 20, 100, 9))
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.huber_function
                )
        Sigma = layer.forward(X, delta=0.4, beta=2)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_backward_tyler(self):
        X = torch.randn((50, 20, 100, 9))
        X.requires_grad = True
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.tyler_function 
                )
        Sigma = layer.forward(X, n_features=9)
        loss = torch.einsum('...ii->', Sigma)
        loss.backward()
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.shape, X.grad.shape)

    def test_backward_student(self):
        X = torch.randn((50, 20, 100, 9))
        X.requires_grad = True
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.student_function
                )
        Sigma = layer.forward(X, n_features=9, nu=2)
        loss = torch.einsum('...ii->', Sigma)
        loss.backward()
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.shape, X.grad.shape)

    def test_backward_huber(self):
        X = torch.randn((50, 20, 100, 9))
        X.requires_grad = True
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.huber_function
                )
        Sigma = layer.forward(X, delta=0.4, beta=2)
        loss = torch.einsum('...ii->', Sigma)
        loss.backward()
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.shape, X.grad.shape)

    def test_assume_centered(self):
        X = torch.randn((50, 20, 100, 9))
        X.requires_grad = True
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.huber_function,
                   assume_centered=True
                )
        Sigma = layer.forward(X, delta=0.4, beta=2)
        loss = torch.einsum('...ii->', Sigma)
        loss.backward()
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.shape, X.grad.shape)

    def test_normalize_trace(self):
        X = torch.randn((50, 20, 100, 9))
        X.requires_grad = True
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.tyler_function,
                   normalize=estimation.normalize_trace
                )
        Sigma = layer.forward(X, n_features=9)
        loss = torch.einsum('...ii->', Sigma)
        loss.backward()
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.shape, X.grad.shape)

    def test_normalize_det(self):
        X = torch.randn((50, 20, 100, 9))
        X.requires_grad = True
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.tyler_function,
                   normalize=estimation.normalize_determinant
                )
        Sigma = layer.forward(X, n_features=9)
        loss = torch.einsum('...ii->', Sigma)
        loss.backward()
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.shape, X.grad.shape)


class TestSCMLayer(TestCase):

    def test_init(self):
        estimation.SCM()

    def test_forward(self):
        layer = estimation.SCM()
        X = torch.randn((50, 20, 100, 9))
        Sigma = layer.forward(X) 
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_backward(self):
        layer = estimation.SCM()
        X = torch.randn((50, 20, 100, 9))
        X.requires_grad = True
        Sigma = layer.forward(X)
        loss = torch.einsum('...ii->', Sigma)
        loss.backward()
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.shape, X.grad.shape)


if __name__ == "__main__":
    main()
