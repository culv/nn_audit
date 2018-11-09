"""
Author: Culver McWhirter
Date:   27 Sep 2018
Time:   15:30
"""

import torch
from torchvision import datasets, transforms
import numpy as np

from sklearn import neighbors
from sklearn.externals import joblib

import h5py

from tqdm import tqdm

import sys
import os

def gather_neuron_data(model, dataset, fname, num_neurons, batch_size=64, compression=None, metadata=dict()):
	"""Collect fully-connected neuron activations from a network for every image in a 
	given dataset, and saves in an HDF5 file. The default dataset fields are 'labels', 'predictions',
	and 'activations'. Other dataset fields may be specified in the 'metadata' input

	Args:
		model = PyTorch model (must be set up to return fully-connected layer activations)
		dataset = A PyTorch Dataset object (contains data and labels)
		fname = Full path to save HDF5 file to
		num_neurons = Number of neurons in hidden fully-connected layers
		batch_size = Size of batches to pass through neural network
		compression = Compression mode for h5py
		metadata = Python dictionary of other datasetfields to save in the HDF5 file (e.g. image shear
			parameters, the image index in the training or test set, the image itself, etc.) 
	"""

	# Get length of training set
	size = len(dataset)

	# Create HDF5 file
	with h5py.File(fname, 'w') as f:
		# Create datasets
		f.create_dataset('labels', (size,), compression=compression, dtype='i')
		f.create_dataset('predictions', (size,), compression=compression, dtype='i')
		f.create_dataset('activations', (size, num_neurons), compression=compression, dtype='f')

		# Create and fill datasets for the fields in other_dict
		for k, v in metadata.items():
			f.create_dataset(k, data=v)

	# Set model to eval mode
	model.eval()

	# Run through training set
	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
	i = 0
	for images, labels in tqdm(iter(loader)):
		# Image batch shape
		bs, c, h, w = images.shape

		# Forward pass of network to get neuron activations
		fc, out = model.forward(images)
	
		# Get network's classification
		predicts = torch.argmax(out, 1).numpy()

		# Save to HDF5 file as separate small datasets
		with h5py.File(fname, 'a') as f:
			f['labels'][i:i+bs] = labels.numpy()
			f['predictions'][i:i+bs] = predicts
			f['activations'][i:i+bs] = fc.detach().numpy()			

		# Increment index
		i += bs



class kNN(object):
	def __init__(self, filename, k=500):
		"""Object for k-nearest neighbors model

		Args:
			filename = The path to save/load the kNN model to/from
			k = Number of nearest neighbors
		"""
		self.filename = filename
		self.knn = neighbors.NearestNeighbors(n_neighbors = k)

	def fit(self, data):
		"""Fit a kNN model with k neighbors to the given data, and save the model to filename using
		scikit-learn's joblib function"""

		# If training points are 1-dim, add axes to make it a 2D array            
		if data.shape[-1] == 1:
			data = data.reshape(1,-1).T

		# Fit kNN to the training data
		self.knn.fit(data)

	def save(self):
		""" Save kNN model to filename"""
		joblib.dump(self.knn, self.filename)

	def load(self):
		"""Load a kNN model from the given filename"""
		self.knn = joblib.load(self.filename)

	def query(self, query_point):
		"""Given a query point, find the nearest neighbor images in the training set and their
		corresponding metadata (labels,	predictions, shears, etc.)

		Args:
			query_point = The query point

		Returns:
			neighbor_idcs = The indices of the neighbors in the training set
			distances = The distances between query point and neighbors
		"""

		# If query point is 1d vector, wrap it to be 2d
		if len(query_point.shape) == 1:
			query_point = query_point[np.newaxis,:]

		# Get the distances and indices of top k nearest neighbors in the train set
		distances, neighbor_idcs = self.knn.kneighbors(query_point)

		# Get rid of singleton dimension
		neighbor_idcs = np.squeeze(neighbor_idcs)

		return neighbor_idcs, distances


def find_training_neighbors(query_point, training_points, training_images, training_labels, k):
	"""Given a query point, find the nearest neighbor images and corresponding labels in the training set

	Args:
		query_point = The query point
		training_points = A NumPy array of all the training points (in the reduced dimension space)
		k = Number of neighbors to find
		training_images = A NumPy array of all the raw training images
		training_labels = A NumPy array of all the training labels

	Returns:
		nearest_images = The top k nearest images
		nearest_labels = The top k nearest labels
	"""

	# If training points are 1-dim, add axes to make it a 2D array            
	if test_point.shape[-1] == 1:
		training_points = training_points.reshape(1,-1).T

	# Fit kNN to the training data
	knn = neighbors.NearestNeighbors(n_neighbors = k)
	knn.fit(training_points)

	# Get the distances and indices of top k nearest neighbors in the train set
	dist, neighbs = knn.kneighbors(test_point[np.newaxis,:])

	# Get rid of singleton dimension
	neighbs = np.squeeze(neighbs)

	# Get nearest images and their labels
	nearest_images = training_images[neighbs]
	nearest_labels = training_labels[neighbs]

	return nearest_images, nearest_labels