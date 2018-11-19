import numpy as np

import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

import sys
import os

# Temporarily add parent directory to path
sys.path.append('./..')
from audit.visual_utils import draw_vector2d

def main():
	# Seed for reproducible results
	np.random.seed(123)

	# Make samples for 3 different classes
	# Covariance matrix, mean, number of samples
	V = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [-np.sqrt(2)/2, np.sqrt(2)/2]])
	D = np.array([[6, 0], [0, 1]])
	cov1 = V.T @ D @ V
	mu1 = np.array([-3, 0])
	n = 100

	X1 = np.random.multivariate_normal(mu1, cov1, n)

	V = np.array([[np.sqrt(3)/2, 1/2], [-1/2, np.sqrt(3)/2]])
	D = np.array([[5, 0], [0, 2]])
	cov2 = V.T @ D @ V
	mu2 = np.array([2.2, -0.2])

	X2 = np.random.multivariate_normal(mu2, cov2, n)


	# Calculate scatter
	mu = (mu1+mu2)/2
	S = (X1-mu).T @ (X1-mu) + (X2-mu).T @ (X2-mu)

	# Do LDA
	D, V = np.linalg.eig(S)

	# Vector for LDA direction
	pca_dir1 = np.array([0, 0, 20*V[0,0], 20*V[1,0]])
	pca_dir2 = np.array([0, 0, -20*V[0,0], -20*V[1,0]])

	sns.set()

	# Original data
	fig, ax = plt.subplots()
	ax.scatter(X1[:,0], X1[:,1], s=35, c='c', edgecolors='k')
	ax.scatter(X2[:,0], X2[:,1], s=35, c='c', edgecolors='k')

	ax.set_aspect('equal')
	ax.set_xlim([-10, 10])
	ax.set_ylim(ax.get_xlim())
	ax.set_yticklabels([])
	ax.set_xticklabels([])

	plt.savefig('pca1.png', bbox_inches='tight')

	# PCA Projection
	fig, ax = plt.subplots()
	alpha = 0.3
	ax.scatter(X1[:,0], X1[:,1], s=35, c='c', edgecolors='k', alpha=alpha, zorder=2)
	ax.scatter(X2[:,0], X2[:,1], s=35, c='c', edgecolors='k', alpha=alpha, zorder=2)

	# Draw line for LDA projection direction
	kw = dict(width=0.003, cmap=plt.cm.gray)
	draw_vector2d(ax, pca_dir1, '0.3', kw)
	draw_vector2d(ax, pca_dir2, '0.3', kw)

	# For every 10th data point, draw a projection line and the
	# projected point
	X1_ = (V[:,0][:,None] @ (X1 @ V[:,0])[None,:]).T
	X2_ = (V[:,0][:,None] @ (X2 @ V[:,0])[None,:]).T

	ax.scatter(X1_[:,0], X1_[:,1], s=35, c='c', edgecolors='k', zorder=2)
	ax.scatter(X2_[:,0], X2_[:,1], s=35, c='c', edgecolors='k', zorder=2)

	kw = dict(width=0.0005, cmap=plt.cm.gray, zorder=1)
	for i in range(n):
		draw_vector2d(ax, [X1[i,0], X1[i,1], X1_[i,0]-X1[i,0], X1_[i,1]-X1[i,1]], '0.2', kw)
		draw_vector2d(ax, [X2[i,0], X2[i,1], X2_[i,0]-X2[i,0], X2_[i,1]-X2[i,1]], '0.2', kw)

	ax.set_aspect('equal')
	ax.set_xlim([-10, 10])
	ax.set_ylim(ax.get_xlim())
	ax.set_yticklabels([])
	ax.set_xticklabels([])

	plt.savefig('pca2.png', bbox_inches='tight')


	plt.show()


if __name__ == '__main__':
	main()