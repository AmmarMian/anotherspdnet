# Script to  check gradients for eigenvalues operations. Version towards input.

import torch
from torch.autograd import gradcheck

from anotherspdnet.functions import (
        ReEigFunction, LogEigFunction, eig_operation,
        eig_operation_gradient_eigs, eig_operation_gradient
)

from geoopt.manifolds import SymmetricPositiveDefinite

from torch.autograd import Function


def operation(x):
    return x**3

def grad_operation(x):
    return 3*x**2


class EigFunction(Function):

    @staticmethod
    def forward(ctx, X):
        eigvals, eigvecs, result = eig_operation(X, operation=operation)
        ctx.save_for_backward(eigvals, eigvecs)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        eigvals, eigvecs = ctx.saved_tensors
        return eig_operation_gradient(
                grad_output, eigvals, eigvecs, operation=operation,
                grad_operation=grad_operation)


def python_eigfunction_trace(X):
    Y = EigFunction.apply(X)
    return torch.einsum('...ii->', Y)


if __name__ == "__main__":

    n_features = 50
    seed = 5555
    batch_size = (3,)
    torch.manual_seed(seed)
    X = SymmetricPositiveDefinite().random(
            batch_size + (n_features, n_features), dtype=torch.float64,
            )
    X.requires_grad = True


    # Computing mnually to see if grad_check works well
    eigvals, eigvecs, Y = eig_operation(X, operation=operation)
    loss = torch.einsum('...ii->', Y)
    print("Automatic gradient")
    loss.backward()
    print(X.grad)

    
    print("Manual gradient")
    grad_output = torch.eye(n_features, dtype=torch.float64).reshape(
        tuple([1 for _ in range(len(batch_size))]) +\
                (n_features, n_features)).repeat(batch_size + (1, 1))
    manual_grad_X = eig_operation_gradient(
            grad_output, eigvals, eigvecs, operation=operation,
            grad_operation=grad_operation)

    print(manual_grad_X)
    print("Difference")
    print(manual_grad_X - X.grad)
    print("Norm of difference")
    print(torch.norm(manual_grad_X - X.grad))


    gradcheck(python_eigfunction_trace, inputs=X)


