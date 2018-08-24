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

import sys


class SimpleNN(nn.module):
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
		x = F.ReLU(x)
		fc_neuron_acts.append(x)

		x = self.fc2(x)
		out = F.sigmoid(x)
		fc_neuron_acts.append(out)

		return fc_neuron_acts, out



def main():
	pass


if __name__ == '__main__':
	main()