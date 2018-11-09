import numpy as np
from scipy.linalg import eig
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
        """Run LDA on data by solving eigendecomposition on Sb*V=Sw*V*D where Sw is the within-class scatter
        matrix and Sb is the between-class scatter matrix. Note that Sw must be full rank for this to be possible.
        To ensure this, PCA is done on the data first and any irrelevation principal components are removed.

        Args:
            X = Data to undergo dimensionality reduction, with shape [n, d] where n
                is the number of samples and d is the dimension
            y = Class labels for data
        """

        # Convert to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        # Shape of data
        n, d = X.shape

        # Convert to float64
        X = X.astype(np.float64)

        # Approximate the rank
        rank = np.linalg.matrix_rank(X)

        # Do PCA first if necessary
        if rank < min(n, d):
            #  Compute mean
            mu = np.mean(X, 0)
            # Zero-out meanm scale by square root of n
            X0 = np.sqrt(n)*(X - mu)
            # Truncated SVD
            _, S, VT = np.linalg.svd(X0, full_matrices=False)
            # Project X onto relevant principal components
            V_PCA = VT.T
            X = X @ V_PCA[:, :rank]
            # Update d
            d = rank
        else:
            V_PCA = np.eye(d)

        # Calculate the priors for each class
        classes = np.unique(y)
        n_classes = classes.shape[0]
        priors = np.bincount(y)/n

        # Find the mean and center data
        mu = np.mean(X, 0)

        # Compute between-class scatter matrix
        Sb = np.zeros([d, d])
        Sw = np.zeros([d, d])

        for c in classes:
            # Get data in class c
            Xc = X[y==c, :]

            # Get class c mean
            mu_c = np.mean(Xc, 0)

            # Add class c component to scatters
            Sb += priors[c]*np.outer(mu_c-mu, mu_c-mu)
            Sw += priors[c]*(Xc-mu_c).T @ Xc-mu_c


        # Do eigendecomposition on inv(Sw)*Sb (i.e. solve the generalized eigenproblem Sb*V=Sw*V*D)
        D, V = eig(Sb, Sw)#np.linalg.pinv(Sw) @ Sb)

        # There will likely be imaginary parts even though Sb and Sw are symmetric; this is due to 
        # precision errors, and the imaginary parts should be nearly zero
        if max(np.imag(D)) >= 1e-5:
            print('Something went wrong, eigenvalues are complex')
            sys.exit()
        else:
            D = np.real(D)

        # Sort eigenvalues and eigenvectors in descending order and truncate since only the first
        # n_classes-1 eigenvalues are nonzero in LDA
        self.D = D[np.argsort(-D)][0:n_classes-1]
        self.V = V[:, np.argsort(-D)][:, 0:n_classes-1]

        # Factor PCA into the projection
        self.V = V_PCA @ self.V

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

    def project(self, X, p=np.inf):
        """Return the data projected onto the top p principal components"""

        # If p is larger than the columns of V, choose the smaller of the two
        p_ = min(p, self.V.shape[1])

        # Subtract mu and project onto first p_ columns of V
        return np.array(X) @ self.V[:,0:p_]


class PCA(object):
    def __init__(self, filename):
        """Object for dimensionality reduction using PCA (principal component analysis) (the truncated SVD
        method is used)"""
        self.filename = filename

        self.V = None
        self.D = None

    def run(self, X):
        """Run truncated SVD on data (similar to PCA but more robust to numerical
        precision and more efficient)

        Args:
            X = Data to undergo dimensionality reduction, with shape [n, d] where n
                is the number of samples and d is the dimension
        """

        # Convert to NumPy array
        X = np.array(X).astype(np.float64)

        # Shape of the data
        n, d = X.shape

        # Zero out the mean of the data and scale by sqrt(n) so that X0'*X0 would give
        # the covariance matrix
        X0 = np.sqrt(n)*(X - np.mean(X, 0))

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
        f.create_dataset('V', data=self.V)
        f.create_dataset('D', data=self.D)

    def load(self):
        """Load the eigenvectors and eigenvalues from filename"""
        f = h5py.File(self.filename, 'r')
        self.V = np.array(f['V'])
        self.D = np.array(f['D'])

    def project(self, X, p=np.inf):
        """Return the data projected onto the top p principal components"""
        # If p is larger than the columns of V, choose the smaller of the two
        p_ = min(p, self.V.shape[1])

        # Subtract mu and project onto first p_ columns of V (first p_ principal components)
        return np.array(X).astype(np.float64) @ self.V[:,0:p_]


def main():
    f = h5py.File('./datasets/neurons/shear_mnist_train_100neurons.hdf5', 'r')

    data = f['activations']#[0:1000]
    labels = f['labels']#[0:1000]

    lda = LDA('test')
    lda.run(data, labels)
    proj = lda.project(data[1,:], p=8)

    pca = PCA('test')
    pca.run(data)
    proj = pca.project(data[1,:], p=20)

if __name__ == '__main__':
    main()