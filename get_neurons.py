"""
Author: Culver McWhirter
Date:   09 Sep 2018
Time:   12:17
"""

import torch

from torchvision import datasets, transforms

import numpy as np

import h5py

from tqdm import tqdm

from one_layer_pytorch import SimpleNN

import sys
import os

def gather_neuron_data(model, train_set, fname, compression=None):
	"""Collect fully-connected neuron activations from a network for every image in a 
	given dataset

	Args:
		model = PyTorch model
		train_set = A PyTorch Dataset object
		fname = Full path to save HDF5 file to
		compression = Compression mode for h5py

	Returns:

	"""
	
	# Get length of training set
	train_size = len(train_set)

	# Create HDF5 Data file
	f = h5py.File(fname, 'w')

	# Create datasets for labels, predictions, and neuron activations
	label_dset = f.create_dataset('labels', (train_size, ), compression=compression, dtype='i')
	predict_dset = f.create_dataset('predictions', (train_size, ), compression=compression, dtype='i')
	neuron_dset = f.create_dataset('activations', (train_size, 100), compression=compression, dtype='f')

	# Set model to eval mode
	model.eval()

	# Run through training set
	i = 0
	for image, label in tqdm(train_set):

		# Reshape image from (1,28,28) to (784,)
		image = image.reshape(image.shape[1]*image.shape[2])

		# Forward pass of network to get neuron activations
		fc_neurons, out = model.forward(image)

		# Get network's classification
		predict = torch.argmax(out).item()

		# Convert hidden fully-connected neurons to NumPy array
		acts = fc_neurons[0].detach().numpy()

		# Save to HDF5
		label_dset[i] = label
		predict_dset[i] = predict
		neuron_dset[i] = acts

		# Increment index
		i += 1


def main():
	# Batch size
	batch_size = 64

	# Path to model
	model_path = './models/fc100_09.pt'

	# Initialize blank model, then load in parameters
	fc = SimpleNN()
	fc.load_state_dict(torch.load(model_path))

	# Get the MNIST training set
	train_data = datasets.MNIST('./datasets/mnist', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
				]))

	gather_neuron_data(fc, train_data, './datasets/neurons/compressed_mnist_train_100neurons.hdf5', compression='gzip')

	# Get the MNIST test set
	test_data = datasets.MNIST('./datasets/mnist', train=False, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
				]))

	gather_neuron_data(fc, test_data, './datasets/neurons/compressed_mnist_test_100neurons.hdf5', compression='gzip')


if __name__ == '__main__':
	main()