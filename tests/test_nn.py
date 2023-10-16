# Test of another_spdnet.nn

from anotherspdnet import nn
from unittest import TestCase, main

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
        self.n_batches = 3
        self.n_matrices = 5
        self.n_in_decrease = 17
        self.n_out_decrease = 7
        self.n_in_increase = 7
        self.n_out_increase = 17

    def test_init(self) -> None:
        """ Test the initialization of the BiMap layer """
        layer = nn.BiMap(self.n_batches, self.n_in_decrease,
                            self.n_out_decrease)
        self.assertEqual(layer.n_in, self.n_in_decrease)
        self.assertEqual(layer.n_out, self.n_out_decrease)
        self.assertEqual(layer.W.shape, (self.n_batches, self.n_out_decrease,
                                        self.n_in_decrease))

    def test_forward_decrease(self) -> None:
        """ Test the forward pass of the BiMap layer.
        Vesion decrease."""
        layer = nn.BiMap(self.n_batches, self.n_in_decrease,
                        self.n_out_decrease)
        X = SPDMatrices(self.n_in_decrease).random_point(
                n_samples=self.n_matrices*self.n_batches)
        X = X.reshape(self.n_batches, self.n_matrices, self.n_in_decrease,
                    self.n_in_decrease)

        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches, self.n_matrices,
                                self.n_out_decrease, self.n_out_decrease))

    def test_forward_increase(self) -> None:
        """ Test the forward pass of the BiMap layer.
        Vesion increase."""
        layer = nn.BiMap(self.n_batches, self.n_in_increase,
                    self.n_out_increase)
        X = SPDMatrices(self.n_in_increase).random_point(
            n_samples=self.n_matrices*self.n_batches)
        X = X.reshape(self.n_batches, self.n_matrices, self.n_in_increase,
                self.n_in_increase)
        Y = layer(X)
        self.assertEqual(Y.shape, (self.n_batches, self.n_matrices,
                            self.n_out_increase, self.n_out_increase))

    def test_backward_decrease(self) -> None:
        """ Test the backward pass of the BiMap layer.
        Vesion decrease."""
        layer = nn.BiMap(self.n_batches, self.n_in_decrease,
                    self.n_out_decrease)
        X = SPDMatrices(self.n_in_decrease).random_point(
            n_samples=self.n_matrices*self.n_batches)
        X = X.reshape(self.n_batches, self.n_matrices, self.n_in_decrease,
                self.n_in_decrease)
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(X.grad.shape, (self.n_batches, self.n_matrices,
                                self.n_in_decrease, self.n_in_decrease))

    def test_backward_increase(self) -> None:
        """ Test the backward pass of the BiMap layer.
        Vesion increase."""
        layer = nn.BiMap(self.n_batches, self.n_in_increase,
                self.n_out_increase)
        X = SPDMatrices(self.n_in_increase).random_point(
            n_samples=self.n_matrices*self.n_batches)
        X = X.reshape(self.n_batches, self.n_matrices, self.n_in_increase,
            self.n_in_increase)
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(X.grad.shape, (self.n_batches, self.n_matrices,
                            self.n_in_increase, self.n_in_increase))

    def test_init_gpu(self) -> None:
        """Test when cuda is available the initialization of the BiMap layer.
        Do nothing if cuda is not available."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            layer = nn.BiMap(self.n_batches, self.n_in_decrease,
                            self.n_out_decrease, device=device)
            self.assertEqual(layer.n_in, self.n_in_decrease)
            self.assertEqual(layer.n_out, self.n_out_decrease)
            self.assertEqual(layer.W.shape, (self.n_batches,
                                        self.n_out_decrease,
                                        self.n_in_decrease))


if __name__ == '__main__':
    main()
