# Test of another_spdnet.parameters

from anotherspdnet import parameters
from unittest import TestCase, main

import torch
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.stiefel import Stiefel

seed = 7777
torch.manual_seed(seed)


class TestStiefelParameter(TestCase):
    """ Test the StiefelParameter """

    def test_decrease_dim(self):
        """Test initialization of StiefelParameter with decreasing 
        dimensions"""
        n_in = 70
        n_out = 30
        W = Stiefel(n_in, n_out).random_uniform(1)
        W = parameters.StiefelParameter(W, verify_stiefel=True)
        self.assertEqual(W.shape, (n_in, n_out))

    def test_increase_dim(self):
        """Test initialization of StiefelParameter with increasing 
        dimensions"""
        n_in = 30
        n_out = 70
        W = Stiefel(n_out, n_in).random_uniform(1)
        W = parameters.StiefelParameter(W, verify_stiefel=True)
        self.assertEqual(W.shape, (n_out, n_in))

    def test_decrease_dim_batches(self):
        """Test initialization of StiefelParameter with decreasing dimensions 
        and batches"""
        n_in = 70
        n_out = 30
        n_batches = 5
        W = Stiefel(n_in, n_out).random_uniform(n_batches)
        W = parameters.StiefelParameter(W, verify_stiefel=True)
        self.assertEqual(W.shape, (n_batches, n_in, n_out))

    def test_increase_dim_batches(self):
        """Test initialization of StiefelParameter with increasing 
        dimensions and batches"""
        n_in = 30
        n_out = 70
        n_batches = 5
        W = Stiefel(n_out, n_in).random_uniform(n_batches)
        W = parameters.StiefelParameter(W, verify_stiefel=True)
        self.assertEqual(W.shape, (n_batches, n_out, n_in))

    def test_getmanifold_increase(self):
        """Test get_manifold method of StiefelParameter with 
        increasing dimensions"""
        n_in = 30
        n_out = 70
        n_batches = 5
        W = Stiefel(n_out, n_in).random_uniform(n_batches)
        W = parameters.StiefelParameter(W, verify_stiefel=True)
        manifold = W.get_manifold()
        assert isinstance(manifold, Stiefel)

    def test_getmanifold_decrease(self):
        """Test get_manifold method of StiefelParameter with
        decreasing dimensions"""
        n_in = 70
        n_out = 30
        n_batches = 5
        W = Stiefel(n_in, n_out).random_uniform(n_batches)
        W = parameters.StiefelParameter(W, verify_stiefel=True)
        manifold = W.get_manifold()
        assert isinstance(manifold, Stiefel)



if __name__ == '__main__':
    main()
