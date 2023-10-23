# Test of another_spdnet.optimizers

from anotherspdnet import optimizers, parameters
from unittest import TestCase, main

from math import prod
import torch
from torch.testing import assert_close
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.stiefel import Stiefel

seed = 7777
torch.manual_seed(seed)


class Test_get_lr_with_strategy(TestCase):
    """Testing the _get_lr_with_strategy function"""
    def test_None(self):
        """Testing the None strategy"""
        lr = 1e-2
        strategy = None
        step_count = 100
        decay = 0
        assert_close(optimizers._get_lr_with_strategy(lr, strategy,
                                                  step_count, decay),
                 lr)

    def test_inverse(self):
        """Testing the inverse strategy"""
        lr = 1e-2
        strategy = "inverse"
        step_count = 100
        decay = 1e-3
        assert_close(optimizers._get_lr_with_strategy(lr, strategy,
                                              step_count, decay),
             lr / (1 + decay * step_count))

    def test_inverse_sqrt(self):
        """Testing the inverse_sqrt strategy"""
        lr = 1e-2
        strategy = "inverse_sqrt"
        step_count = 100
        decay = 1e-3
        assert_close(optimizers._get_lr_with_strategy(lr, strategy,
                                          step_count, decay),
             lr / (1 + decay * step_count)**0.5)

    def test_exponential(self):
        """Testing the exponential strategy"""
        lr = 1e-2
        strategy = "exponential"
        step_count = 100
        decay = 0.9
        assert_close(optimizers._get_lr_with_strategy(lr, strategy,
                                          step_count, decay),
             lr * decay**step_count)

    def test_linear(self):
        """Testing the linear strategy"""
        lr = 1e-2
        strategy = "linear"
        step_count = 100
        decay = 1e-3
        assert_close(optimizers._get_lr_with_strategy(lr, strategy,
                                      step_count, decay),
             lr * (1 - decay * step_count))

    
class TestManifoldGradientDescent(TestCase):
    """Testing the ManifoldGradientDescent optimizer"""

    def setUp(self) -> None:
        self.manifold = Stiefel(5, 3)
        self.n_batches = (2, 3, 4)
        W = self.manifold.random_point(n_samples=prod(self.n_batches))
        W = W.reshape(self.n_batches + W.shape[-2:])
        self.stiefel_param = parameters.StiefelParameter(
                W, requires_grad=True, verify_stiefel=True)

    def test_init(self):
        """Testing the constructor"""
        optimizer = optimizers.ManifoldGradientDescent(
                [self.stiefel_param], [self.manifold])
        assert optimizer.lr == 1e-2
        assert optimizer.parameters == [self.stiefel_param]
        assert optimizer.manifolds == [self.manifold]

        optimizer = optimizers.ManifoldGradientDescent(
                [self.stiefel_param], lr=1e-3)
        assert optimizer.lr == 1e-3
        assert optimizer.parameters == [self.stiefel_param]
        assert isinstance(optimizer.manifolds[0], Stiefel)

        def must_fail():
            _ = optimizers.ManifoldGradientDescent(
                    [self.stiefel_param], [self.manifold, self.manifold])
        self.assertRaises(AssertionError, must_fail)

    def test_step(self):
        """Testing the step method"""
        param_stiefel_before = self.stiefel_param.data.clone()
        optimizer = optimizers.ManifoldGradientDescent(
            [self.stiefel_param], [self.manifold])
        optimizer.zero_grad()
        loss = torch.sum(self.stiefel_param**2)
        loss.backward()
        optimizer.step()
        assert_close(self.stiefel_param.data,
                 self.manifold.metric.exp(-optimizer.lr *
                                          self.stiefel_param.grad,
                                        param_stiefel_before))

    def test_zero_grad(self):
        """Testing the zero_grad method"""
        optimizer = optimizers.ManifoldGradientDescent(
            [self.stiefel_param], [self.manifold])
        optimizer.zero_grad()
        assert self.stiefel_param.grad is None
        loss = torch.sum(self.stiefel_param**2)
        loss.backward()
        assert self.stiefel_param.grad is not None
        optimizer.zero_grad()
        assert_close(self.stiefel_param.grad,
                    torch.zeros_like(self.stiefel_param.grad))

    def test_repr(self):
        """Test representation of the optimizer"""
        optimizer = optimizers.ManifoldGradientDescent(
            [self.stiefel_param], [self.manifold])
        assert isinstance(optimizer.__repr__(), str)

    def test_str(self):
        """Test string representation of the optimizer"""
        optimizer = optimizers.ManifoldGradientDescent(
            [self.stiefel_param], [self.manifold])
        assert isinstance(optimizer.__str__(), str)


class TestMixRiemannianOptimizer(TestCase):
    """Testing the MixRiemannianOptimizer optimizer"""

    def setUp(self) -> None:
        self.manifold = Stiefel(5, 3)
        self.n_batches = (2, 3, 4)
        W = self.manifold.random_point(n_samples=prod(self.n_batches))
        W = W.reshape(self.n_batches + W.shape[-2:])
        self.stiefel_param = parameters.StiefelParameter(
            W, requires_grad=True, verify_stiefel=True)
        self.stiefel_param2 = parameters.StiefelParameter(
                W, requires_grad=True, verify_stiefel=True)
        self.standard_param = torch.nn.Parameter(torch.randn(2, 3, 4))

    def test_init(self):
        """Testing the constructor"""
        optimizer = optimizers.MixRiemannianOptimizer(
                [self.stiefel_param, self.stiefel_param2,
                 self.standard_param], torch.optim.Adam,
                lr=1e-3)
        assert optimizer.lr == 1e-3
        assert optimizer.manifold_parameters == [self.stiefel_param,
                                                self.stiefel_param2]
        assert optimizer.standard_parameters == [self.standard_param]
        assert isinstance(optimizer.manifold_optimizer,
                        optimizers.ManifoldGradientDescent)
        assert isinstance(optimizer.standard_optimizer, torch.optim.Adam)

        optimizer = optimizers.MixRiemannianOptimizer(
                [self.stiefel_param, self.stiefel_param2], torch.optim.Adam,
                lr=1e-3)
        assert optimizer.standard_optimizer is None

    def test_step(self):
        """Testing the step method"""
        param_stiefel_before = self.stiefel_param.data.clone()
        param_stiefel2_before = self.stiefel_param2.data.clone()
        optimizer = optimizers.MixRiemannianOptimizer(
            [self.stiefel_param, self.stiefel_param2,
             self.standard_param], torch.optim.Adam,
            lr=1e-3)
        optimizer.zero_grad()
        loss = torch.sum(self.stiefel_param**2) + \
               torch.sum(self.stiefel_param2**2) + \
               torch.sum(self.standard_param**2)
        loss.backward()
        optimizer.step()
        assert_close(self.stiefel_param.data,
             self.manifold.metric.exp(-optimizer.lr *
                                      self.stiefel_param.grad,
                                    param_stiefel_before))
        assert_close(self.stiefel_param2.data,
             self.manifold.metric.exp(-optimizer.lr *
                                      self.stiefel_param2.grad,
                                    param_stiefel2_before))

    def test_zero_grad(self):
        """Testing the zero_grad method"""
        optimizer = optimizers.MixRiemannianOptimizer(
            [self.stiefel_param, self.stiefel_param2,
             self.standard_param], torch.optim.Adam,
            lr=1e-3)
        optimizer.zero_grad()
        assert self.stiefel_param.grad is None
        assert self.stiefel_param2.grad is None
        assert self.standard_param.grad is None
        loss = torch.sum(self.stiefel_param**2) + \
               torch.sum(self.stiefel_param2**2) + \
               torch.sum(self.standard_param**2)
        loss.backward()
        assert self.stiefel_param.grad is not None
        assert self.stiefel_param2.grad is not None
        assert self.standard_param.grad is not None
        optimizer.zero_grad()
        assert_close(self.stiefel_param.grad,
                torch.zeros_like(self.stiefel_param.grad))
        assert_close(self.stiefel_param2.grad,
                torch.zeros_like(self.stiefel_param2.grad))
        assert self.standard_param.grad is None


    def test_repr(self):
        """Test representation of the optimizer"""
        optimizer = optimizers.MixRiemannianOptimizer(
                [self.stiefel_param, self.stiefel_param2,
                 self.standard_param], torch.optim.Adam,
                lr=1e-3)
        assert isinstance(optimizer.__repr__(), str)

    def test_str(self):
        """Test string representation of the optimizer"""
        optimizer = optimizers.MixRiemannianOptimizer(
            [self.stiefel_param, self.stiefel_param2,
             self.standard_param], torch.optim.Adam,
            lr=1e-3)
        assert isinstance(optimizer.__str__(), str)
