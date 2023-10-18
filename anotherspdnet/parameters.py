# ========================================
# FileName: parameters.py
# Date: 11 october 2023 - 14:04
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# Brief: Parameters of torch models
# =========================================

import torch
from torch import nn

class StiefelParameter(nn.Parameter):
    """Parameter belonging in the manifold of Stiefel matrices.
    i.e :math:`\\mathcal{S}t_{n,k} = \\{\\mathbf{M}\\in\\mathcal{M}_{n,k}: 
    \\mathbf{M}^{\\mathrm{T}}\\mathbf{M}=\\mathbf{I}_k\\}`.
    """

    def __new__(cls, data, requires_grad=True,
                verify_stiefel=False) -> nn.Parameter:
        """Create a new StiefelParameter with some data that can
        also be verified to be a Stiefel matrix.

        Parameters
        ----------
        data : torch.Tensor
            Data of the parameter.

        requires_grad : bool, optional
            Whether the parameter requires gradient. Default is True.

        verify_stiefel : bool, optional
            Whether to verify that the data is a Stiefel matrix.
            Default is False.

        Returns
        -------
        torch.nn.Parameter
            The created StiefelParameter.
        """
        if data is not None and verify_stiefel:
            assert data.shape[-2] >= data.shape[-1], \
                    "Stiefel matrices must have more rows than columns"
            if data.ndim == 2:
                assert torch.allclose(
                        data.T @ data, torch.eye(data.shape[-1])), \
                    "Stiefel matrices must be orthogonal"
            else:
                multidim_eye = torch.eye(data.shape[-1]).reshape(
                            (1,)*(data.ndim-2) + (data.shape[-1], data.shape[-1])
                        ).repeat(data.shape[:-2] + (1, 1))
                assert torch.allclose(
                        torch.bmm(data.transpose(-2, -1), data),
                        multidim_eye), \
                    "Stiefel matrices must be orthogonal"
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        """Representation of the StiefelParameter.

        Returns
        -------
        str
            Representation of the StiefelParameter.
        """
        string = f'StiefelParameter'
        if self.requires_grad:
            string += '(requires_grad, '
        string+= f'shape={self.shape})'
        return string
