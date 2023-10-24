# Script to  check gradients for eigenvalues operations. Version towards
# eigenvalues and eigenvectors.

import torch
from torch.autograd import gradcheck

from anotherspdnet.functions import (
        ReEigFunction, LogEigFunction, eig_operation,
        eig_operation_gradient_eigs, eig_operation_gradient
)

from geoopt.manifolds import SymmetricPositiveDefinite

from torch.autograd import Function


def operation(x):
    return x**4

def grad_operation(x):
    return 4*x**3


class EigFunction(Function):

    @staticmethod
    def forward(ctx, eigvals, eigvecs):
        ctx.save_for_backward(eigvals, eigvecs)
        _eigvals = torch.diag_embed(operation(eigvals))
        return torch.einsum(
                '...ij,...jk,...kl->...il',
                eigvecs, _eigvals, eigvecs.transpose(-1, -2))


    @staticmethod
    def backward(ctx, grad_output):
        eigvals, eigvecs = ctx.saved_tensors
        grad_eigvals, grad_eigvecs = eig_operation_gradient_eigs(
                grad_output, eigvals, eigvecs, operation=operation,
                grad_operation=grad_operation)
        return torch.diagonal(grad_eigvals, dim1=-2, dim2=-1), \
                grad_eigvecs



def python_eigfunction_trace(eigvals, eigvecs):
    Y = EigFunction.apply(eigvals, eigvecs)
    return torch.einsum('...ii->', Y)


if __name__ == "__main__":

    n_features = 70
    seed = 5555
    batch_size = (4,)
    torch.manual_seed(seed)
    X = SymmetricPositiveDefinite().random(
            batch_size + (n_features, n_features), dtype=torch.float64,
            )
    eigvals, eigvecs, _ = eig_operation(X, operation=lambda x: x)
    eigvals.requires_grad_(True)
    eigvecs.requires_grad_(True)
    
    # Computing mnually to see if grad_check works well
    Y = torch.einsum('...ij,...jk,...kl->...il', eigvecs, torch.diag_embed(
        operation(eigvals)), eigvecs.transpose(-1, -2))
    loss = torch.einsum('...ii->', Y)
    loss.backward()

    
    grad_output = torch.eye(n_features, dtype=torch.float64).reshape(
        tuple([1 for _ in range(len(batch_size))]) +\
                (n_features, n_features)).repeat(batch_size + (1, 1))
    manual_grad_eigvals, manual_grad_eigvecs = eig_operation_gradient_eigs(
            grad_output, eigvals, eigvecs, operation=operation,
            grad_operation=grad_operation)
    manual_grad_eigvals = torch.diagonal(manual_grad_eigvals, dim1=-2, dim2=-1)

    print("Manual gradient eigvals")
    print(manual_grad_eigvals)
    print("Automatic gradient eigvals")
    print(eigvals.grad)
    print("Difference")
    print(manual_grad_eigvals - eigvals.grad)

    print("\n\n")
    print("Manual gradient eigvecs")
    print(manual_grad_eigvecs)
    print("Automatic gradient eigvecs")
    print(eigvecs.grad)
    print("Difference")
    print(manual_grad_eigvecs - eigvecs.grad)

    gradcheck(python_eigfunction_trace, inputs=(eigvals, eigvecs))


