# Test of another_spdnet.nn

from anotherspdnet import nn
from unittest import TestCase, main

import torch
from torch.testing import assert_close

from geoopt.manifolds import SymmetricPositiveDefinite

seed = 7777
torch.manual_seed(seed)

# Change default torch to float64
torch.set_default_dtype(torch.float64)


# =============================================================================
# BiMap layer
# =============================================================================
class TestBiMap(TestCase):
    """Test the BiMap layer"""

    def setUp(self) -> None:
        self.n_batches = (3,)
        self.n_batches_many = (3, 2, 5, 7)
        self.n_matrices = 5
        self.n_in_decrease = 17
        self.n_out_decrease = 7
        self.n_in_increase = 7
        self.n_out_increase = 17

    def test_init(self) -> None:
        """Test the initialization of the BiMap layer"""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, self.n_batches)
        self.assertEqual(layer.n_in, self.n_in_decrease)
        self.assertEqual(layer.n_out, self.n_out_decrease)
        self.assertEqual(
            layer.W.shape, (self.n_batches + (self.n_in_decrease, self.n_out_decrease))
        )

    def test_repr(self) -> None:
        """Test the representation of the BiMap layer"""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, self.n_batches)
        assert isinstance(repr(layer), str)

    def test_str(self) -> None:
        """Test the string representation of the BiMap layer"""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, self.n_batches)
        assert isinstance(str(layer), str)

    def test_forward_decrease(self) -> None:
        """Test the forward pass of the BiMap layer.
        Version decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, self.n_batches)
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_in_decrease, self.n_in_decrease),
            dtype=torch.float64,
        )

        Y = layer(X)
        self.assertEqual(
            Y.shape,
            (
                self.n_batches
                + (self.n_matrices, self.n_out_decrease, self.n_out_decrease)
            ),
        )

    def test_forward_increase(self) -> None:
        """Test the forward pass of the BiMap layer.
        Version increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase, self.n_batches)
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_in_increase, self.n_in_increase),
            dtype=torch.float64,
        )
        Y = layer(X)
        self.assertEqual(
            Y.shape,
            (
                self.n_batches
                + (self.n_matrices, self.n_out_increase, self.n_out_increase)
            ),
        )

    def test_backward_decrease(self) -> None:
        """Test the backward pass of the BiMap layer.
        Vesion decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, self.n_batches)
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_in_decrease, self.n_in_decrease),
            dtype=torch.float64,
        )
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(
            X.grad.shape,
            (
                self.n_batches
                + (self.n_matrices, self.n_in_decrease, self.n_in_decrease)
            ),
        )

    def test_backward_increase(self) -> None:
        """Test the backward pass of the BiMap layer.
        Version increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase, self.n_batches)
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_in_increase, self.n_in_increase),
            dtype=torch.float64,
        )
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(
            X.grad.shape,
            (
                self.n_batches
                + (self.n_matrices, self.n_in_increase, self.n_in_increase)
            ),
        )

    def test_init_gpu(self) -> None:
        """Test when cuda is available the initialization of the BiMap layer.
        Do nothing if cuda is not available."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            layer = nn.BiMap(
                self.n_in_decrease, self.n_out_decrease, self.n_batches, device=device
            )
            self.assertEqual(layer.n_in, self.n_in_decrease)
            self.assertEqual(layer.n_out, self.n_out_decrease)
            self.assertEqual(
                layer.W.shape,
                (self.n_batches + (self.n_in_decrease, self.n_out_decrease)),
            )

    def test_forward_decrease_many(self) -> None:
        """Test the forward pass of the BiMap layer with many batches
        dimensions. Version decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, self.n_batches_many)

        X = SymmetricPositiveDefinite().random(
            self.n_batches_many
            + (self.n_matrices, self.n_in_decrease, self.n_in_decrease),
            dtype=torch.float64,
        )
        Y = layer(X)
        self.assertEqual(
            Y.shape,
            (
                self.n_batches_many
                + (self.n_matrices, self.n_out_decrease, self.n_out_decrease)
            ),
        )

    def test_forward_increase_many(self) -> None:
        """Test the forward pass of the BiMap layer with many batches
        dimensions. Version increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase, self.n_batches_many)
        X = SymmetricPositiveDefinite().random(
            self.n_batches_many
            + (self.n_matrices, self.n_in_increase, self.n_in_increase),
            dtype=torch.float64,
        )
        Y = layer(X)
        self.assertEqual(
            Y.shape,
            (
                self.n_batches_many
                + (self.n_matrices, self.n_out_increase, self.n_out_increase)
            ),
        )

    def test_backward_decrease_many(self) -> None:
        """Test the backward pass of the BiMap layer with many batches
        dimensions. Version decrease."""
        layer = nn.BiMap(self.n_in_decrease, self.n_out_decrease, self.n_batches_many)
        X = SymmetricPositiveDefinite().random(
            self.n_batches_many
            + (self.n_matrices, self.n_in_decrease, self.n_in_decrease),
            dtype=torch.float64,
        )
        X.requires_grad = True
        Y = layer(X)
        Y.sum().backward()
        self.assertEqual(
            X.grad.shape,
            (
                self.n_batches_many
                + (self.n_matrices, self.n_in_decrease, self.n_in_decrease)
            ),
        )

    def test_backward_increase_many(self) -> None:
        """Test the backward pass of the BiMap layer with many batches
        dimensions. Vesion increase."""
        layer = nn.BiMap(self.n_in_increase, self.n_out_increase, self.n_batches_many)
        X = SymmetricPositiveDefinite().random(
            self.n_batches_many
            + (self.n_matrices, self.n_in_increase, self.n_in_increase),
            dtype=torch.float64,
        )
        X.requires_grad = True
        Y = layer(X)
        loss = torch.einsum("...ii->", Y)
        loss.backward()
        self.assertEqual(
            X.grad.shape,
            (
                self.n_batches_many
                + (self.n_matrices, self.n_in_increase, self.n_in_increase)
            ),
        )
        assert_close(X.grad, X.grad.transpose(-1, -2))

    def test_deviceisrespected(self) -> None:
        """Test if the device is respected when initializing the BiMap layer"""
        # Cuda if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            layer = nn.BiMap(
                self.n_in_decrease, self.n_out_decrease, self.n_batches, device=device
            )
            self.assertEqual(layer.W.device.type, device.type)

            X = (
                SymmetricPositiveDefinite()
                .random(
                    self.n_batches
                    + (self.n_matrices, self.n_in_decrease, self.n_in_decrease),
                    dtype=torch.float64,
                )
                .to(device)
            )
            Y = layer(X)
            self.assertEqual(Y.device.type, device.type)


# =============================================================================
# Test of the ReEig layer
# =============================================================================
class TestReEig(TestCase):
    """Test the ReEig layer"""

    def setUp(self) -> None:
        self.n_batches = (3,)
        self.n_batches_many = (3, 2, 5, 7)
        self.n_matrices = 5
        self.n_features = 7
        self.eps = 1e-4

    def test_init(self) -> None:
        """Test the initialization of the ReEig layer"""
        layer = nn.ReEig(self.eps)
        self.assertEqual(layer.eps, self.eps)

        _ = nn.ReEig(self.eps, use_autograd=True)

    def test_repr(self) -> None:
        """Test the representation of the ReEig layer"""
        layer = nn.ReEig(self.eps)
        assert isinstance(repr(layer), str)

    def test_str(self) -> None:
        """Test the string representation of the ReEig layer"""
        layer = nn.ReEig(self.eps)
        assert isinstance(str(layer), str)

    def test_forward(self) -> None:
        """Test the forward pass of the ReEig layer"""
        layer = nn.ReEig(self.eps)
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_features, self.n_features),
            dtype=torch.float64,
        )
        Y = layer(X)
        self.assertEqual(
            Y.shape,
            (self.n_batches + (self.n_matrices, self.n_features, self.n_features)),
        )

    def test_backward(self) -> None:
        """Test the backward pass of the ReEig layer"""
        layer = nn.ReEig(self.eps)
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_features, self.n_features),
            dtype=torch.float64,
        )
        X.requires_grad = True
        Y = layer(X)
        loss = torch.einsum("...ii->", Y)
        loss.backward()
        self.assertEqual(
            X.grad.shape,
            (self.n_batches + (self.n_matrices, self.n_features, self.n_features)),
        )

    def test_deviceisrespected(self) -> None:
        """Test if the device is respected for ReEig layer"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            layer = nn.ReEig(self.eps)
            X = (
                SymmetricPositiveDefinite()
                .random(
                    self.n_batches + (self.n_matrices, self.n_features, self.n_features)
                )
                .to(device)
            )
            X.requires_grad = True
            Y = layer(X)
            assert Y.device.type == device.type


# =============================================================================
# LogEig layer
# =============================================================================
class TestLogEig(TestCase):
    """Test the LogEig layer"""

    def setUp(self) -> None:
        self.n_batches = (3,)
        self.n_batches_many = (3, 2, 5, 7)
        self.n_matrices = 5
        self.n_features = 7
        self.eps = 1e-4

    def test_init(self) -> None:
        """Test the initialization of the LogEig layer"""
        _ = nn.LogEig()
        _ = nn.LogEig(use_autograd=True)

    def test_repr(self) -> None:
        """Test the representation of the LogEig layer"""
        layer = nn.LogEig()
        assert isinstance(repr(layer), str)

    def test_str(self) -> None:
        """Test the string representation of the LogEig layer"""
        layer = nn.LogEig()
        assert isinstance(str(layer), str)

    def test_forward(self) -> None:
        """Test the forward pass of the ReEig layer"""
        layer = nn.LogEig()
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_features, self.n_features)
        )
        Y = layer(X)
        self.assertEqual(
            Y.shape,
            (self.n_batches + (self.n_matrices, self.n_features, self.n_features)),
        )

    def test_backward(self) -> None:
        """Test the backward pass of the LogEig layer"""
        layer = nn.LogEig()
        X = SymmetricPositiveDefinite().random(
            self.n_batches + (self.n_matrices, self.n_features, self.n_features)
        )
        X.requires_grad = True
        Y = layer(X)
        loss = torch.einsum("...ii->", Y)
        loss.backward()
        self.assertEqual(
            X.grad.shape,
            (self.n_batches + (self.n_matrices, self.n_features, self.n_features)),
        )


# =============================================================================
# Vectorization layers
# =============================================================================
class TestVectorization(TestCase):
    """Testing Vectorization layer"""

    def setUp(self) -> None:
        self.data = torch.randn(2, 3, 7, 4, 5)
        self.data.requires_grad = True

    def test_init(self) -> None:
        """Test the initialization of the Vectorization layer"""
        _ = nn.Vectorization()

    def test_forward(self) -> None:
        """Test the forward pass of the Vectorization layer"""
        layer = nn.Vectorization()
        y = layer(self.data)
        assert y.shape == (2, 3, 7, 20)

    def test_inverse_transfrom(self) -> None:
        """Test the inverse transform of the Vectorization layer"""
        layer = nn.Vectorization()
        y = layer(self.data)
        inv_y = layer.inverse_transform(y, self.data.shape[-2])
        assert inv_y.shape == self.data.shape
        assert inv_y.dtype == self.data.dtype


class TestVech(TestCase):
    """Testing Vech layer"""

    def setUp(self) -> None:
        self.data = torch.randn(2, 3, 7, 5, 5)
        self.data.requires_grad = True

    def test_init(self) -> None:
        """Test the initialization of the Vech layer"""
        _ = nn.Vech()

    def test_forward(self) -> None:
        """Test the forward pass of the Vech layer"""
        layer = nn.Vech()
        y = layer(self.data)
        assert y.shape == (2, 3, 7, 15)

    def test_inverse_transfrom(self) -> None:
        """Test the inverse transform of the Vech layer"""
        layer = nn.Vech()
        y = layer(self.data)
        inv_y = layer.inverse_transform(y)
        assert inv_y.shape == self.data.shape
        assert inv_y.dtype == self.data.dtype


if __name__ == "__main__":
    main()
