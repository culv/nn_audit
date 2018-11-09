"""
Author: Culver McWhirter
Date:   7 Nov 2018
Time:   14:15
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import numpy as np

import argparse

from tqdm import tqdm

import sys
import os


# Command line arguments
parser = argparse.ArgumentParser(description='Pytorch ConvNet for MNIST')

parser.add_argument('--batch-size', type=int, default=64,
	help='input batch size for testing/training (default: 64)')

parser.add_argument('--epochs', type=int, default=10,
	help='number of epochs to train (default: 10)')

parser.add_argument('--save-name', type=str, required=True,
	help='base filename to save models to (required)')

parser.add_argument('--shear', action='store_true',
	help='use Pytorch random shear transform (default: False)')


class ConvNet(nn.Module):
	"""A simple Convolutional Neural Network for MNIST"""
	def __init__(self):
		super(ConvNet, self).__init__()

		self.conv_layer1 = nn.Sequential(
			nn.Conv2d(1, 32, 5, stride=1, padding=2), # outputs 32 feature maps
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2))

		self.conv_layer2 = nn.Sequential(
			nn.Conv2d(32, 64, 5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2))

		self.dropout_layer = nn.Dropout()

		self.fc_layer1 = nn.Sequential(
			nn.Linear(64*7*7, 512),
			nn.ReLU())

		self.fc_layer2 = nn.Sequential(
			nn.Linear(512, 10))


	def forward(self, images):
		x = self.conv_layer1(images)
		x = self.conv_layer2(x)
		x = x.view(x.shape[0], -1)
		x = self.dropout_layer(x)
		fc = self.fc_layer1(x)
		out = self.fc_layer2(fc)

		return fc, out


def save_model(model, epoch, save_header):
	"""Saves neural networks to {script directory}/{save_header}_{epoch}.pt"""

	# Get path to this script's directory
	path = os.path.dirname(os.path.realpath(__file__))
	# Construct full save path
	save_to = os.path.join(path, '{}_{:02d}.pt'.format(save_header, epoch))
	# Save
	torch.save(model.state_dict(), save_to)


def eval_model(model, test_loader):
	"""Evaluate a model on the training set"""

	# Keep track of correct count
	correct = 0

	# Set to evaluation mode
	model.eval()

	# Iterate over all test data
	for images, labels in tqdm(iter(test_loader)):

		# Forward pass
		fc, out = model.forward(images)

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
		datasets.MNIST('.././datasets/mnist', train=True, download=True,
			transform=xfm), batch_size=args.batch_size, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('.././datasets/mnist', train=False, download=False,
			transform=xfm), batch_size=args.batch_size, shuffle=True)

	# Create model
	cnn = ConvNet()

	# Use Adam optimizer and cross entropy loss
	optimizer = optim.Adam(cnn.parameters())
	criterion = nn.CrossEntropyLoss()


	# Training loop
	for epoch in range(args.epochs):

		# Keep track of correct count
		correct = 0

		for images, labels in tqdm(iter(train_loader)):

			# Forward pass
			fc, out = cnn.forward(images)
			loss = criterion(out, labels)

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Calculate predicted answers
			pred = out.data.max(1)[1]
			correct += pred.eq(labels.data).sum()

		# Compute train accuracy
		train_acc = float(correct.item()) / len(train_loader.dataset)

		# Evaluate on test data
		test_acc = eval_model(cnn, test_loader)

		# Give progress update
		print('[Epoch {}/{}] Train Acc: {:5.2f} Test Acc: {:5.2f}'.format(epoch, args.epochs-1, 100*train_acc, 100*test_acc))

		# Save the model
		save_model(cnn, epoch, args.save_name)


if __name__ == '__main__':
	main()