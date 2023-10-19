# Test of anotherspdnet.utils module


from anotherspdnet import utils
from unittest import main

from math import prod
import torch
from torch.testing import assert_close
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.spd_matrices import SPDMatrices

seed = 55555
torch.manual_seed(seed)

def test_symmetrize():
    """Test the symmetrize function."""
    n_features = 17
    n_batches = (3, 2, 40)
    X = torch.randn(n_batches + (n_features, n_features))
    sym_X = utils.symmetrize(X)
    assert sym_X.shape == X.shape
    assert_close(sym_X, sym_X.transpose(-1, -2))
    assert sym_X.dtype == X.dtype

def test_construct_eigdiff_matrix():
    """Test the construct_eigdiff_matrix function."""
    n_features = 17
    n_batches = (3, 2, 40)
    X = SPDMatrices(n=n_features).random_point(prod(n_batches))
    X = X.reshape(n_batches + (n_features, n_features))
    eigvals = torch.linalg.eigvalsh(X)
    eigdiff_matrix = utils.construct_eigdiff_matrix(eigvals)
    
    assert eigdiff_matrix.shape == X.shape
    assert_close(eigdiff_matrix.diagonal(dim1=-2, dim2=-1),
                 torch.zeros_like(eigdiff_matrix.diagonal(dim1=-2, dim2=-1)))
    assert eigdiff_matrix.dtype == X.dtype
    assert torch.allclose(eigdiff_matrix, -eigdiff_matrix.transpose(-1, -2))


def test_zero_offdiag():
    X = torch.randn(3, 2, 40, 17, 17)
    X_zero = utils.zero_offdiag(X)
    assert X_zero.shape == X.shape
    assert_close(X_zero.diagonal(dim1=-2, dim2=-1), X.diagonal(dim1=-2, dim2=-1))
    assert X_zero.dtype == X.dtype
    assert_close(X_zero, X_zero.transpose(-1, -2))


if __name__ == "__main__":
    main()


