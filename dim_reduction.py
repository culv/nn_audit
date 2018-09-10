import numpy as np

from sklearn import decomposition

import h5py

import os
import sys

def truncated_SVD(data):
    """Run truncated SVD on data (similar to PCA but more robust to numerical
    precision and more efficient)

    Args:
        data = Data to undergo dimensionality reduction

    Returns:
        evecs = Ordered eigenvectors (used to transform data)
        evals = Ordered eigenvalues
    """

    # Initialize the truncated SVD class, with n_components as the largest possible
    svd = decomposition.TruncatedSVD(n_components = data.shape[-1]-1)

    # Zero out the mean of the data
    data = data - np.mean(data, 0)

    # Run truncated SVD on the zero-mean data
    svd.fit(data)

    # Get the basis of eigenvectors, and their accompanying eigenvalues (singular values squared)
    evecs = svd.components_.T
    evals = svd.singular_values_**2

    return evecs, evals


def main():
    f = h5py.File('./datasets/neurons/compressed_mnist_train_100neurons.hdf5', 'r')

    data = f['activations']

    evecs, evals = truncated_SVD(data)

    print(evals)

if __name__ == '__main__':
    main()