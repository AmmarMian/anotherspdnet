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


class TestMestimationLayer(TestCase):

    def test_init(self):
        """Test instanciation of Mestimation layer"""
        estimation.Mestimation(lambda x: 1/x)

    def test_forward_tyler(self):
        """Test forward Tyler"""
        X = torch.randn((50, 20, 100, 9))
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.tyler_function 
                )
        Sigma = layer.forward(X, n_features=9)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_forward_student(self):
        """Test forward Student-t"""
        X = torch.randn((50, 20, 100, 9))
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.student_function
                )
        Sigma = layer.forward(X, n_features=9, nu=2)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_forward_huber(self):
        """Test forward Huber"""
        X = torch.randn((50, 20, 100, 9))
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.huber_function
                )
        Sigma = layer.forward(X, delta=0.4, beta=2)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_backward_tyler(self):
        """Test backward Tyler"""
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
        """Test Backward Student-t"""
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
        """Test backward Huber"""
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
        """Test Mestimation when assume centered data"""
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
        """Test Tyler normalized by trace"""
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
        """Test Tyler normalized by determinant"""
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


    def test_estimation_init2D(self):
        """Test Mestimation with an initial 2D value"""
        X = torch.randn((50, 20, 100, 9))
        init = torch.cov(torch.randn(9, 500))
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.tyler_function 
                )
        Sigma = layer.forward(X, init=init, n_features=9)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_estimation_initBatch(self):
        """Test Mestimation with an inital batch value"""
        X = torch.randn((50, 20, 100, 9))
        init = estimation.SCM().forward(X)
        layer = estimation.Mestimation(
                   m_estimation_function=estimation.tyler_function 
                )
        Sigma = layer.forward(X, init=init, n_features=9)
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

class TestSCMLayer(TestCase):

    def test_init(self):
        """Test instanciation of SCM layer"""
        estimation.SCM()

    def test_forward(self):
        """Test forward of SCM layer"""
        layer = estimation.SCM()
        X = torch.randn((50, 20, 100, 9))
        Sigma = layer.forward(X) 
        self.assertEqual(Sigma.shape, (50, 20, 9, 9))
        assert_close(Sigma, Sigma.transpose(-2, -1))

    def test_backward(self):
        """Test backward of SCM layer"""
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
