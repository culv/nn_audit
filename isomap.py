from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import axes3d
from sklearn import (manifold, datasets)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST/', one_hot=True)

digits = datasets.load_digits(n_class = 3)
X = digits.data
y = digits.target

# dataset_name = 'mnist_validation'

# X = np.loadtxt('mnist_100neurons_data.gz')
# print(X.shape)
# y = np.argmax(mnist.validation.labels, 1)
# print(y.shape)

n_samples, n_features = X.shape
n_neighbors = 30
emb_dim = 2

# scale and visualize the embedding vectors
# INPUTS: 	X = the embedding
#			title = desired title of the plot
#			plot_type = either plot embedded vectors as 'scatter' or 'digit'
def plot_embedding(X  , title=None, plot_type = 'scatter'):
	# normalize all embedding vectors
	x_min, x_max = np.min(X,0), np.max(X,0)
	X = (X - x_min) / (x_max - x_min)

	# get dimension of embedding vectors
	dim = X.shape[1]

	if dim == 3:
		fig = plt.figure(dpi=60)
		ax = axes3d.Axes3D(fig) #fig.gca(projection='3d')
		#ax.set_axis_off()

		# loop over all vectors in embedding (same as # of digits)
		if plot_type == 'digit':
			for i in range(X.shape[0]):
				# plot digit as a string, at the location determined by the embedding X
				ax.text(X[i,0], X[i,1], X[i,2], str(y[i]),
					color=plt.cm.tab10(y[i] / 10.), # color determined from Set1 color map
					fontdict={'weight': 'bold', 'size': 9}) # format font
	
		elif plot_type == 'scatter':
			ax.scatter(X.T[0,:], X.T[1,:], X.T[2,:], c = y)



	if dim == 2:
		# create 2d figure
		plt.figure()
		ax = plt.subplot(111)

		# loop over all vectors in embedding (same as # of digits)
		if plot_type == 'digit':
			for i in range(X.shape[0]):
				# plot digit as a string, at the location determined by the embedding X
				plt.text(X[i,0], X[i,1], str(digits.target[i]), ###### WHY
					color=plt.cm.tab10(y[i] / 10.), # color determined from Set1 color map
					fontdict={'weight': 'bold', 'size': 9}) # format font
		elif plot_type == 'scatter':
			plt.scatter(X.T[0,:], X.T[1,:], c=y)

		# show digit images on plot
		if hasattr(offsetbox,  'AnnotationBbox'): # will only work with matplotlib versions past v1.0
			# initialize shown images locations array with upper right corner of plot
			shown_images = np.array([[1.,1.]])
			# loop over all digits
			for i in range(digits.data.shape[0]):
				# calculate squared distance between current image's embedding vector and all others that have been displayed
				dist = np.sum((X[i] - shown_images) ** 2, 1)
				# if the smallest squared distance is below threshold, don't print it (to ensure plot isn't overcrowded)
				if np.min(dist) < 4e-3:
					continue
				# otherwise, add embedding vector to array of shown images
				shown_images = np.r_[shown_images, [X[i]]]
				# and display image of digit at the embedding vector location
				imagebox = offsetbox.AnnotationBbox(
					offsetbox.OffsetImage(digits.images[i],  cmap=plt.cm.gray_r),
					X[i])
				ax.add_artist(imagebox)
	# no tick marks
	plt.xticks([]), plt.yticks([])
	# set title if given
	if title is not None:
		plt.title(title)


# n_img_per_row = 20
# img = np.zeros((10*n_img_per_row, 10*n_img_per_row))
# for i in range(n_img_per_row):
# 	ix = 10 * i + 1
# 	for j in range(n_img_per_row):
# 		iy = 10*j + 1
# 		img[ix:ix+8, iy:iy+8] = X[i*n_img_per_row+j].reshape((8,8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')


print('Computing ISOMAP embedding')
t0  = time()
X_iso = manifold.Isomap(n_neighbors, n_components=emb_dim).fit_transform(X)
print('Done.')
plot_embedding(X_iso, 'ISOMAP projection of the digits (time %.2fs)'%(time()-t0), 'digit')

# save the embedding vectors
# isomap_embedding_fn = '{}dim_isomap_{}.gz'.format(emb_dim, dataset_name)
# np.savetxt(isomap_embedding_fn, X_iso)

plt.show()