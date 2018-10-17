import numpy as np

from sklearn import decomposition

import h5py

import os
import sys


class PCA(object):
    def __init__(self, filename, p):
        """Object for dimensionality reduction using PCA/truncated SVD"""
        self.filename = filename
        self.p = p

        self.svd = None
        self.V = None
        self.D = None

    def run(self, data):
        """Run truncated SVD on data (similar to PCA but more robust to numerical
        precision and more efficient)

        Args:
            data = Data to undergo dimensionality reduction
        """

        # Initialize the truncated SVD class, with n_components as the largest possible
        self.svd = decomposition.TruncatedSVD(n_components = data.shape[-1]-1)

        # Zero out the mean of the data
        data = data - np.mean(data, 0)

        # Run truncated SVD on the zero-mean data
        self.svd.fit(data)

        # Get the basis of eigenvectors, and their accompanying eigenvalues (singular values squared)
        self.V = self.svd.components_.T
        self.D = self.svd.singular_values_**2

    def save(self):
        """Save the eigenvectors and eigenvalues in HDF5 file at filename"""
        f = h5py.File(self.filename, 'w')
        f.create_dataset('V', data=self.V)
        f.create_dataset('D', data=self.D)

    def load(self):
        """Load the eigenvectors and eigenvalues from filename"""
        f = h5py.File(self.filename, 'r')
        self.V = f['V']
        self.D = f['D']

    def project(self, data):
        """Return the data projected onto the top p principal components"""
        return np.matmul(data, self.V[:,0:self.p])


def main():
    f = h5py.File('./datasets/neurons/compressed_mnist_train_100neurons.hdf5', 'r')

    data = f['activations']

    evecs, evals = truncated_SVD(data)

    print(evals)

if __name__ == '__main__':
    main()