import numpy as np
from scipy.linalg import eig
from scipy.spatial import KDTree
import h5py

import os
import sys

from tqdm import tqdm, trange

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
			V_PCA = VT.T[:, :rank]
			X = X @ V_PCA
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

		# Zero out the mean of the data and scale by 1/sqrt(n) so that X0'*X0 would give
		# the covariance matrix
		X0 = (X - np.mean(X, 0))/np.sqrt(n)

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


class DML(object):
	def __init__(self, filename):
		"""Object for distance metric learning (DML) using minibatch stochastic gradient
		descent (SGD). A minibatch SGD method is used to find the linear transform"""
		self.filename = filename

		self.M = None

	def run(self, X, y, n_targets=3, n_imposters=3, batch=256, step=0.01, hinge_L=8):
		"""Run truncated SVD on data (similar to PCA but more robust to numerical
		precision and more efficient)

		Args:
			X = Data to undergo dimensionality reduction, with shape [n, d] where n
				is the number of samples and d is the dimension
			y = Corresponding labels
			k = Number of neighbors with same label to consider "inside" margin
			batch = Batch size for SGD (default=264)
		"""
		# Cast to NumPy arrays
		X = np.array(X).astype(np.float64)
		y = np.array(y).astype(np.int)

		# Shape of data
		n, d = X.shape

		# Get triplet constraints
		tri = self.triplets(X, y, n_targets=n_targets, n_imposters=n_imposters)

		# Get hinge loss
		h = smooth_hinge(L=hinge_L)

		# Initialize M
		M = []
		M.append(np.eye(d))

		# Compute the number of batches
		T = np.ceil(len(tri)/batch).astype(np.int)
		tri_ids = np.random.choice(len(tri), len(tri), replace=False)

		# Iterate over batches
		print('Doing gradient descent...')
		for t in trange(T):
			# Sum gradient over batch
			grad = 0
			for s in tri_ids[t*batch:(t+1)*batch]:
				i, j, k = tri[s]
				delta_ij = (X[i]-X[j])[:, None]
				delta_ik = (X[i]-X[k])[:, None]

				h_deriv = h.grad((delta_ij.T.dot(M[t])*delta_ij.T).sum(axis=1) - (delta_ik.T.dot(M[t])*delta_ik.T).sum(axis=1) + 1)
				grad += h_deriv/batch*(np.outer(delta_ij, delta_ij) - np.outer(delta_ik, delta_ik))

			# Update
			M.append(self.project_feasible(M[t] - step*grad))

		# Compute solution
		self.M = sum(M)/T

		return self.M

	def triplets(self, X, y, n_targets=3, n_imposters=3):
		"""Get set of triplet constraints. Do this by finding target neighbors and imposters using
		kd trees

		Args:
			X = Data matrix with shape [n, d] (n=number of samples, d=dimension)
			y = Corresponding labels
			n_targets = Number of target neighbors (neighbors with same label) that determine the
				boundary (so the distance to the (n_targets)th neighbor with the same label)
			n_imposters = Number of imposters (neighbors with different label) to consider inside
				boundary
		"""

		# Cast data and labels to NumPy arrays
		X = np.array(X).astype(np.float64)
		y = np.array(y).astype(np.int)

		n, d = X.shape
		leaf = np.ceil(float(n)/10).astype(np.int)
		# Construct kd tree with a large leaf size
		kd = KDTree(X, leafsize=leaf)

		# Loop over samples
		triplets = []
		i = 0
		print('Finding imposters...')
		for x in tqdm(X):
			# Label of x
			l = y[i]

			# Query point to find nearest points, ordered
			distances, neighbors = kd.query(x, k=leaf)

			# Find index of the "target", i.e. the (n_targets)th neighbor with same label. If there
			# aren't that many same-label neighbors in the leaf, just pick farthest neighbor with same label
			same_count = 0
			for j in neighbors:
				if y[j]==l:
					same_count += 1
				if same_count==n_targets:
					break

			# Find "imposters", i.e. neighbors with different labels that are closer than target,
			# limited to n_imposters. Once you find an imposter, append (i,j,k) to triplets
			imposter_count = 0
			for k in neighbors[:j]:
				if y[k]!=l:
					imposter_count += 1
					triplets.append((i,j,k))
				if imposter_count==n_imposters:
					break

			i += 1

		return triplets

	def project_feasible(self, A, R=1000):
		"""Project matrix A onto the feasible set given by {A: A positive semidefinite, frobenius_norm(A)<=R}.
		R default is 1000"""

		A = project_PSD(A)
		fro = np.sum(np.abs(A))
		return min(R/fro, 1)*A

	def save(self):
		"""Save the linear transform in HDF5 file at filename"""
		f = h5py.File(self.filename, 'w')
		f.create_dataset('M', data=self.M)

	def load(self):
		"""Load the linear transform from filename"""
		f = h5py.File(self.filename, 'r')
		self.M = np.array(f['M'])

	def project(self, X):
		"""Return the data projected using linear transform"""

		# Subtract mu and project onto first p_ columns of V (first p_ principal components)
		return np.array(X).astype(np.float64) @ self.M


class smooth_hinge(object):
	def __init__(self, L=8):
		"""Smooth hinge function h(z)=(1/L)*log(1+exp(L*z)) where we choose L>0.
		The larger L is, the closer approximation to the hinge function (default=8)"""
		self.L = L

	def __call__(self, z):
		"""Compute the value h(z)"""
		return (1/self.L)*np.log(1+np.exp(self.L*z))

	def grad(self, z):
		"""Compute the partial derivative dh/dz, evaluated at z. Note that this derivative
		is the sigmoid function, 1/(1+exp(-L*z))"""
#		print(z, 1/(1+np.exp(-self.L*z)))
		return 1/(1+np.exp(-self.L*z))		


def project_PSD(A):
	"""Project a matrix A onto the positive-semidefinite (PSD) cone. This amounts to replacing
	any negative eigenvalues with zero."""
	# Get eigenvalues, D, and eigenvectors, V
	D, V = np.linalg.eig(A)

	# Truncate negative eigenvalues (in-place)
	np.maximum(D, 0, D)

	A_psd = V.dot(np.diag(D)).dot(V.T)

	# Check for complex values
	if np.max(np.imag(A_psd))<=1e-5:
		A_psd = np.real(A_psd)
	else:
		error('Got complex values in matrix')

	return A_psd
	

def main():
	f = h5py.File('./../datasets/neurons/conv_mnist_test.hdf5', 'r')

	data = f['activations']#[0:1000]
	labels = f['predictions']#[0:1000]

	dml = DML('test')
	M = dml.run(data, labels)
	dml.save()

	print(M[100, 0:10])
	print(np.max(M))
	print(np.min(M))
	sys.exit()
	l = smooth_hinge(L=8)
	print(l(-5))
	print(l(5))
	print(l.grad(-5))
	print(l.grad(5))


	lda = LDA('test')
	lda.run(data, labels)
	proj = lda.project(data[1,:], p=8)

	pca = PCA('test')
	pca.run(data)
	proj = pca.project(data[1,:], p=20)

if __name__ == '__main__':
	main()