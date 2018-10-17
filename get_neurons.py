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
import argparse

from one_layer_pytorch import SimpleNN, RandomChoiceShear
from nn_audit_utils import gather_neuron_data

import sys
import os

# Command line arguments
parser = argparse.ArgumentParser(description='Pytorch one layer fully-connected')

parser.add_argument('--batch-size', type=int, default=64,
	help='batch size for neural network input (default=64)')

parser.add_argument('--model-path', type=str, required=True,
	help='relative path (from curent directory) to load Pytorch model from (required)')

parser.add_argument('--save-path', type=str, required=True,
	help='relative path (from current directory) to save neuron data to (required)')

parser.add_argument('--train', action='store_true',
	help='collect neuron data on the train set')

parser.add_argument('--shear', action='store_true',
	help='collect neuron data for the random shear experiment')


def main():
	# Get command line args
	args = parser.parse_args()
	print(args)

	# Initialize blank model, then load in parameters from args.model_path
	fc = SimpleNN()
	fc.load_state_dict(torch.load(args.model_path))

	if args.shear:
		# Get MNIST set for each shear in shear_list
		shear_list = np.linspace(-50, 50, 11).astype(np.int8)

		xfms = [transforms.Compose([
			RandomChoiceShear([shear]),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
			]) for shear in shear_list]

		dsets = [datasets.MNIST('./datasets/mnist', train=args.train, download=True, transform=xfm) for xfm in xfms]
		
		# Concatenate them into one dataset of the form [all MNIST with shear 1, all MNIST with shear 2, ...]
		data = torch.utils.data.ConcatDataset(dsets)

		# Create corresponding list of shears for each image
		shears = [shear for shear in shear_list for i in range(len(dsets[0]))]

		# Create corresponding list of MNIST training set index for each image
		mnist_index = np.tile(np.linspace(0, len(dsets[0])-1, len(dsets[0])), len(dsets)).astype(np.uint)

		# Create dictionary of other metadata (shears and mnist index) to pass to gather_neuron_data()
		metadata_dict = dict(shears=shears, mnist_index=mnist_index)

	else:
		# Get the MNIST set
		xfm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
			])

		data = datasets.MNIST('./datasets/mnist', train=args.train, download=True, transform=xfm)
		
		mnist_index = np.linspace(0, len(data)-1, len(data)).astype(np.uint)
		metadata_dict = dict(mnist_index=mnist_index)


	gather_neuron_data(fc, data, args.save_path, batch_size=args.batch_size, compression='gzip', metadata=metadata_dict)


if __name__ == '__main__':
	main()