"""
Author: Culver McWhirter
Date:   25 Aug 2018
Time:   12:19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import numpy as np

import argparse

from tqdm import tqdm

import sys
import os

# Command line arguments
parser = argparse.ArgumentParser(description='Pytorch one layer fully-connected')

parser.add_argument('--batch-size', type=int, default=64,
	help='input batch size for testing/training (default: 64)')

parser.add_argument('--epochs', type=int, default=10,
	help='number of epochs to train (default: 10)')

parser.add_argument('--save-name', type=str, required=True,
	help='base filename to save models to (required)')

parser.add_argument('--shear', type=bool, default=False,
	help='use Pytorch random shear transform (default: False)')



class SimpleNN(nn.Module):
	"""A simple 2-layer fully-connected network
	Args:
		c_in = Number of inputs (default: 784 for MNIST)
		c_hidden = Number of hidden neurons (default: 100)
		c_out = Number of output neurons (default: 10 for MNIST)
	
	Methods:
		forward(): Forward pass of network
	"""
	def __init__(self, c_in=784, c_hidden=100, c_out=10):
		super(SimpleNN, self).__init__()

		self.fc1 = nn.Linear(c_in, c_hidden)
		self.fc2 = nn.Linear(c_hidden, c_out)

	def forward(self, x):
		"""Forward pass
		Args:
			x = Batch of input data
		Returns:
			out = Predictions
			fc_neuron_acts = Fully-connected neuron activations
		"""

		fc_neuron_acts = []

		x = self.fc1(x)
		x = F.relu(x)
		fc_neuron_acts.append(x)

		x = self.fc2(x)
		out = F.sigmoid(x)
		fc_neuron_acts.append(out)

		return fc_neuron_acts, out


def save_model(model, epoch, fname_base, save_dir='./models'):
	"""Saves neural networks to {save_dir}/{fname_base}_{epoch}.pt"""

	# Create save directory if it doesn't exist
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Full path to save file
	save_to = os.path.join(save_dir, fname_base+'_{:02d}.pt'.format(epoch))

	# Save model
	torch.save(model.state_dict(), save_to)


def eval_model(model, test_loader):
	"""Evaluate a model on the training set"""

	# Keep track of correct count
	correct = 0

	# Set to evaluation mode
	model.eval()

	# Iterate over all test data
	for images, labels in tqdm(iter(test_loader)):

		# Reshape images
		images = images.reshape(images.shape[0], images.shape[2]*images.shape[3])

		# Forward pass
		act, out = model.forward(images)

		# Calculate predicted answers and accuracy of the batch
		pred = out.data.max(1)[1]
		correct += pred.eq(labels.data).sum()

	# Calculate accuracy
	acc = float(correct.item()) / len(test_loader.dataset)
	return acc

class RandomChoiceShear(object):
	"""Custom PyTorch data transform to randomly shear a PIL Image using a shear chosen from
	a preset list"""
	def __init__(self, shear_choices):
		self.shear_choices = shear_choices

	def __call__(self, img):
		shear = np.random.choice(self.shear_choices)
		return transforms.functional.affine(img, angle=0, translate=(0,0), scale=1, shear=shear)


def main():
	# Parse command line arguments/hyperparameters
	args = parser.parse_args()
	print(args)
	
	# Basic transform
	basic_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])

	# Random shear transform
	shear_choices = np.linspace(-50,50,11)
	random_shear_transform = transforms.Compose([
		RandomChoiceShear(shear_choices),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])

	# Transform to use
	if args.shear:
		xfm = random_shear_transform
	else:
		xfm = basic_transform


	# Use MNIST dataset
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./datasets/mnist', train=True, download=True,
			transform=xfm), batch_size=args.batch_size, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./datasets/mnist', train=False, download=False,
			transform=xfm), batch_size=args.batch_size, shuffle=True)

	# Create simple FC model
	fc_net = SimpleNN()

	# Use Adam optimizer and cross entropy loss
	optimizer = optim.Adam(fc_net.parameters())
	criterion = nn.CrossEntropyLoss()


	# Training loop
	for epoch in range(args.epochs):
		for images, labels in tqdm(iter(train_loader)):

			# Zero out gradient buffers
			optimizer.zero_grad()

			# Reshape images (flatten)
			images = images.reshape(images.shape[0], images.shape[2]*images.shape[3])

			# Forward pass of network, collecting both FC activations and output
			fc_neurons, out = fc_net.forward(images)

			# Calculate loss and backpropagate
			loss = criterion(out, labels)
			loss.backward()
			optimizer.step()

			# Calculate predicted answers and accuracy of the batch
			pred = out.data.max(1)[1]
			correct = pred.eq(labels.data).sum()
			acc = float(correct.item()) / labels.shape[0]

		# Evaluate on test data
		test_acc = eval_model(fc_net, test_loader)

		print('[Epoch {}] Train Acc: {:5.2f} Test Acc: {:5.2f}'.format(epoch, 100*acc, 100*test_acc))
		save_model(fc_net, epoch, args.save_name)



if __name__ == '__main__':
	main()