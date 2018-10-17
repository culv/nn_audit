import numpy as np

import h5py

import os
import sys

class LDA(object):
    def __init__(self, filename):
        """Object for dimensionality reduction using LDA (linear discriminant analysis)"""
        self.filename = filename

        self.V = None
        self.D = None

    def run(self, X, y):
        """Run LDA on data

        Args:
            X = Data to undergo dimensionality reduction, with shape [n, d] where n
                is the number of samples and d is the dimension
            y = Class labels for data
        """

        # Shape of data
        n, d = X.shape


        # Calculate the priors for each class
        classes = np.unique(y)
        n_classes = classes.shape[0]
        priors = np.bincount(y)/n

        # Find the mean
        mu = np.mean(X, 0)

        # Compute between-class and within-class scatter
        Sb = np.zeros([d, d])
        Sw = np.zeros([d, d])

        for c in classes:
            # Get data in class c
            Xc = X[y==c, :]

            # Get class c mean
            mu_c = np.mean(Xc, 0)

            # Add class c component to scatters
            Sb += priors[c]*np.outer(mu_c-mu, mu_c-mu)
            Sw += priors[c]*np.matmul((Xc-mu_c).T, Xc-mu_c)

        # Do SVD on A=inv(Sw)*Sb
        A = np.matmul(np.linalg.pinv(Sw), Sb)
        self.V, S, _ = np.linalg.svd(A, full_matrices=False)

        # Truncate V and S since only the first n_classes-1 singular values are nonzero in LDA
        self.V = self.V[:, 0:n_classes-1]
        S = S[0:n_classes-1]

        # Square singular values to get eigenvalues
        self.D = S**2


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

    def project(self, data, p=np.inf):
        """Return the data projected onto the top p principal components"""

        # If p is larger than the columns of V, choose the smaller of the two
        p_ = min(p, self.V.shape[1])

        return np.matmul(data, self.V[:,0:p_])


class PCA(object):
    def __init__(self, filename):
        """Object for dimensionality reduction using PCA/truncated SVD"""
        self.filename = filename

        self.mu = None
        self.V = None
        self.D = None

    def run(self, X):
        """Run truncated SVD on data (similar to PCA but more robust to numerical
        precision and more efficient)

        Args:
            X = Data to undergo dimensionality reduction, with shape [n, d] where n
                is the number of samples and d is the dimension
        """

        # Shape of the data
        n, d = X.shape

        # Zero out the mean of the data and scale by sqrt(n) so that X0'*X0 would give
        # the covariance matrix
        self.mu = np.mean(X, 0)
        X0 = np.sqrt(n)*(X - self.mu)

        # Run truncated SVD on the zero-mean data, only keep nonzero eigenvalues
        # For SVD of X0=USV', the rows of V are eigenvectors of X0'X0
        _, S, VT = np.linalg.svd(X0, full_matrices=False)

        # Convert singular values to eigenvalues (i.e. square them)
        self.D = S**2

        # Transpose eigenvectors so that they are column vectors instead of row vectors
        self.V = VT.T

    def save(self):
        """Save the eigenvectors and eigenvalues in HDF5 file at filename"""
        f = h5py.File(self.filename, 'w')
        f.create_dataset('mu', data=self.mu)
        f.create_dataset('V', data=self.V)
        f.create_dataset('D', data=self.D)

    def load(self):
        """Load the eigenvectors and eigenvalues from filename"""
        f = h5py.File(self.filename, 'r')
        self.mu = np.array(f['mu'])
        self.V = np.array(f['V'])
        self.D = np.array(f['D'])

    def project(self, X, p=np.inf):
        """Return the data projected onto the top p principal components"""
        # If p is larger than the columns of V, choose the smaller of the two
        p_ = min(p, self.V.shape[1])

        # Subtract mu and project onto first p_ columns of V (first p_ principal components)
        return np.matmul(np.array(X)-self.mu, self.V[:,0:p_])


def main():
    f = h5py.File('./datasets/neurons/shear_mnist_train_100neurons.hdf5', 'r')

    data = f['activations'][0:1000]
    labels = f['labels'][0:1000]

    lda = LDA('test')
    lda.run(data, labels)
    proj = lda.project(data[1,:], p=8)

    pca = PCA('test')
    pca.run(data)
    proj = pca.project(data[1,:], p=20)

if __name__ == '__main__':
    main()