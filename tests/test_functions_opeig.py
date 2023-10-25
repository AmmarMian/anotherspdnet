# Test of another_spdnet.functions relative to functions of the eigenvalues

from anotherspdnet import functions
from unittest import TestCase, main, result

from math import prod
import torch
from torch.testing import assert_close

from geoopt.manifolds import SymmetricPositiveDefinite

seed = 7777
torch.manual_seed(seed)

class TestEigOperation(TestCase):
    """Test the function eig_operation"""

    def setUp(self) -> None:
        """Set up the test"""
        self.batch_size = (3, 2, 4)
        self.n_features = 7
        self.X = SymmetricPositiveDefinite().random(
                self.batch_size + (self.n_features, self.n_features))

    def test_identity_eigh(self):
        """Test doing no operation with eigh mode"""
        operation = lambda x: x
        eigvals, eigvects, result = functions.eig_operation(
                self.X, operation, "eigh")
        assert eigvals.shape == self.batch_size + (self.n_features,)
        assert eigvects.shape == self.batch_size + (self.n_features,
                                                    self.n_features)
        assert result.shape == self.X.shape
        assert torch.all(torch.real(eigvals) > 0)
        assert_close(eigvects @ eigvects.transpose(-1, -2),
                    torch.eye(self.n_features).repeat(
                        self.batch_size + (1, 1)))
        assert result.dtype == self.X.dtype

    def test_identity_eig(self):
        """Test doing no operation with eig mode"""
        operation = lambda x: x
        eigvals, eigvects, result = functions.eig_operation(
            self.X, operation, "eig")
        assert eigvals.shape == self.batch_size + (self.n_features,)
        assert eigvects.shape == self.batch_size + (self.n_features,
                                                self.n_features)
        assert result.shape == self.X.shape
        assert torch.all(torch.real(eigvals) > 0)
        assert_close(eigvects @ eigvects.transpose(-1, -2),
                torch.eye(self.n_features).repeat(
                    self.batch_size + (1, 1)))
        assert result.dtype == self.X.dtype

    def test_squared_eig(self):
        """Test squaring the eigenvalues"""
        operation = lambda x: x**2
        eigvals, eigvects, result = functions.eig_operation(
            self.X, operation, "eig")
        assert eigvals.shape == self.batch_size + (self.n_features,)
        assert eigvects.shape == self.batch_size + (self.n_features,
                                            self.n_features)
        assert result.shape == self.X.shape
        assert torch.all(torch.real(eigvals) > 0)
        assert result.dtype == self.X.dtype

        # TODO: Find out if the comparison is reasonable
        # # Compute again the eigenvalues and eigenvectors on the
        # # result
        # eigvals2, eigvects2 = torch.linalg.eig(result)
        # eigvals2, eigvects2 = torch.real(eigvals2), torch.real(eigvects2)

        # # Normalise both eigenvalues to compare them
        # eigvals = torch.sort(eigvals, dim=-1).values
        # eigvals2 = torch.sort(eigvals2, dim=-1).values
        # eigvects = torch.sort(eigvects, dim=-1).values
        # eigvects2 = torch.sort(eigvects2, dim=-1).values

        # assert_close(torch.real(eigvects2), eigvects, rtol=1e-4, atol=1e-4)
        # assert_close(torch.real(eigvals2), eigvals**2, rtol=1e-4, atol=1e-4)


class TestEigOperationGradient(TestCase):
    """Test the functions eig_operation_gradient_eigs and 
    eig_operation_gradient"""

    def setUp(self) -> None:
        """Set up the test"""
        self.batch_size = (3, 2, 4)
        self.n_features = 7
        self.operation = lambda x: x**2
        self.grad_operation = lambda x: 2*x
        self.loss_function = lambda x: torch.einsum('...ii->', x)
        self.grad_output = torch.eye(
                self.n_features).reshape(
                        tuple([1 for _ in range(len(self.batch_size))]) +\
                        (self.n_features, self.n_features)).repeat(
                    self.batch_size + (1, 1)) 

    def test_gradient_eigs(self):
        """Testing the gradient towards eigenvalues and eigenvectors"""
        # Doing forward manually to have autograd
        X = SymmetricPositiveDefinite().random(
                self.batch_size + (self.n_features, self.n_features))
        eigvals, eigvects = torch.linalg.eig(X)
        eigvals, eigvects = torch.real(eigvals), torch.real(eigvects)
        eigvals.requires_grad = True
        eigvects.requires_grad = True
        Y = torch.einsum('...ij,...jk,...kl->...il',
                         eigvects, torch.diag_embed(self.operation(eigvals)),
                         eigvects.transpose(-1, -2))
        loss = self.loss_function(Y)
        loss.backward()

        grad_eigvals, grad_eigvects = functions.eig_operation_gradient_eigs(
                self.grad_output, eigvals, eigvects,
                self.operation, self.grad_operation)
        assert grad_eigvals.shape == self.batch_size + (self.n_features,
                                                        self.n_features)
        assert grad_eigvects.shape == self.batch_size + (self.n_features,
                                                        self.n_features)
        assert grad_eigvals.dtype == X.dtype

        grad_eigvals = torch.diagonal(grad_eigvals, dim1=-2, dim2=-1)

        assert_close(grad_eigvals, eigvals.grad)
        assert_close(grad_eigvects, eigvects.grad)


    def test_grad_input(self):
        """Testing the gradient towards input"""
        X = SymmetricPositiveDefinite().random(
                self.batch_size + (self.n_features, self.n_features))
        X.requires_grad = True
        eigvals, eigvects, Y = functions.eig_operation(
                X, self.operation)
        loss = self.loss_function(Y)
        loss.backward()
        grad_X = functions.eig_operation_gradient(
            self.grad_output, eigvals, eigvects,
            self.operation, self.grad_operation)
        assert grad_X.shape == X.shape
        assert grad_X.dtype == X.dtype
        # TODO: FIX THIS SHIT
        # assert_close(grad_X, X.grad)


class TestReEig(TestCase):
    """Test the toch implemented ReEig Function"""
    def setUp(self) -> None:
        """Set up the test"""
        self.batch_size = (3, 2, 4)
        self.n_features = 7
        self.X = SymmetricPositiveDefinite().random(
                self.batch_size + (self.n_features, self.n_features))
        self.X.requires_grad = True
        self.operation = lambda x: functions.re_operation(x, 1e-4)
        self.grad_operation = lambda x: functions.re_operation_gradient(x, 1e-4)
        _, _, Y = functions.eig_operation(self.X, self.operation)
        self.Y = Y
        self.grad_output = torch.eye(
            self.n_features).reshape(
                    1, 1, self.n_features, self.n_features).repeat(
                self.batch_size + (1, 1))
        loss = torch.einsum('...ii->', Y)
        loss.backward()

    def test_forward(self):
        """Test forward pass"""
        Y = functions.ReEigFunction.apply(self.X, 1e-4)
        assert Y.shape == self.X.shape
        assert Y.dtype == self.X.dtype
        assert_close(Y, self.Y)

        
class TestLogEig(TestCase):
    """Test the toch implemented LogEig Function"""

    def setUp(self) -> None:
        """Set up the test"""
        self.batch_size = (3, 2, 4)
        self.n_features = 7
        self.X = SymmetricPositiveDefinite().random(
                self.batch_size + (self.n_features, self.n_features))
        self.X.requires_grad = True
        self.operation = torch.log
        self.grad_operation = lambda x: 1/x
        _, _, Y = functions.eig_operation(self.X, self.operation)
        self.Y = Y
        self.grad_output = torch.eye(
            self.n_features).reshape(
                1, 1, self.n_features, self.n_features).repeat(
            self.batch_size + (1, 1))
        loss = torch.einsum('...ii->', Y)
        loss.backward()

    def test_forward(self):
        """Test forward pass"""
        Y = functions.LogEigFunction.apply(self.X)
        assert Y.shape == self.X.shape
        assert Y.dtype == self.X.dtype
        assert_close(Y, self.Y)


if __name__ == "__main__":
    main()
