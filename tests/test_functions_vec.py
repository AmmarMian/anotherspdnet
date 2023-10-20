# Test of another_spdnet.functions relative to vectorization and 
# unvectorization

from anotherspdnet import functions
from unittest import TestCase, main

import torch


class TestVecUnvec(TestCase):
    def setUp(self) -> None:
        self.data = torch.randn(2, 3, 7, 4, 5)
        self.data_dim2 = torch.randn(5, 4)

    def test_vec_batch(self):
        """Testing vec function"""
        y = functions.vec_batch(self.data)
        y_dim2 = functions.vec_batch(self.data_dim2)
        assert y.shape == (2, 3, 7, 20)
        assert y_dim2.shape == (20,)
        assert y.dtype == self.data.dtype
        assert y_dim2.dtype == self.data_dim2.dtype

    def test_unvec_batch(self):
        """Testing unvec function"""
        y = functions.vec_batch(self.data)
        y_dim2 = functions.vec_batch(self.data_dim2)
        u_y = functions.unvec_batch(y, self.data.shape[-2])
        u_y_dim2 = functions.unvec_batch(y_dim2, self.data_dim2.shape[-2])
        assert u_y.shape == self.data.shape
        assert u_y_dim2.shape == self.data_dim2.shape
        assert u_y.dtype == self.data.dtype
        assert u_y_dim2.dtype == self.data_dim2.dtype


class TestVechUnvechBatch(TestCase):
    def setUp(self) -> None:
        self.data = torch.randn(2, 3, 7, 15, 15)
        self.data2 = torch.randn(2, 3, 7, 14, 14)

    def test_vech_batch(self):
        """Testing vech_batch function"""
        y = functions.vech_batch(self.data)
        y2 = functions.vech_batch(self.data2)
        assert y.shape == (2, 3, 7, 120)
        assert y.dtype == self.data.dtype
        assert y2.shape == (2, 3, 7, 105)
        assert y2.dtype == self.data2.dtype

    def test_unvech_batch(self):
        """Testing unvech_batch function"""
        y = functions.vech_batch(self.data)
        y2 = functions.vech_batch(self.data2)
        u_y = functions.unvech_batch(y)
        u_y2 = functions.unvech_batch(y2)
        assert u_y.shape == self.data.shape
        assert u_y.dtype == self.data.dtype
        assert u_y2.shape == self.data2.shape
        assert u_y2.dtype == self.data2.dtype


if __name__ == '__main__':
    main()
