# Testing some usual Riemannian functiosn on SPD matrices
# Script to  check gradients for eigenvalues operations. Version towards
# eigenvalues and eigenvectors.

import torch

from anotherspdnet.functions import (
    spd_affine_invariant_distance
)

from geoopt.manifolds import SymmetricPositiveDefinite



if __name__ == "__main__":

    n_features = 70
    seed = 5555
    batch_size = (4,)
    torch.manual_seed(seed)
    X = SymmetricPositiveDefinite().random(
            batch_size + (n_features, n_features), dtype=torch.float64,
            )
    Y = SymmetricPositiveDefinite().random(
            batch_size + (n_features, n_features), dtype=torch.float64,
            )

    # Compute the distance
    dist = spd_affine_invariant_distance(X, Y)

