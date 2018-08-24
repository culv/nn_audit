import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import numpy as np

import sys

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()

		self.conv_layer1 = nn.Sequential(
			nn.Conv2d(1, 32, 5, stride=1, padding=0), # outputs 32 feature maps
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2))

		self.conv_layer2 = nn.Sequential(
			nn.Conv2d(32, 64, 5, stride=1, padding=0),
			nn.ReLU(),
			nn.Dropout2d(),
			nn.MaxPool2d(2, stride=2))

		self.fc_layer1 = nn.Sequential(
			nn.Linear(64*4*4, 128),
			nn.ReLU())

		self.fc_layer2 = nn.Sequential(
			nn.Linear(128, 10))

		self.optimizer = optim.Adam(self.parameters())

	def forward(self, image):
		x = self.conv_layer1(image)
		x = self.conv_layer2(x)
		x = x.view(x.shape[0], -1)
		x = self.fc_layer1(x)
		x = self.fc_layer2(x)

		return x

	def train(self, image, label):
		self.optimizer.zero_grad()
		out = self.forward(image)

		criterion = nn.CrossEntropyLoss()
		loss = criterion(out, label)

		loss.backward()

		self.optimizer.step() # backpropagate
		pred = out.data.max(1)[1] # predicted answer
		correct = pred.eq(label.data).cpu().sum() # determine if model was correct (on the CPU)
		acc = correct * 100. / label.shape[0] # calculate accuracy
		return loss, acc


	# save model during training
	def save_model(self, epoch):
		if not os.path.exists('./models'):
			os.makedirs('./models')
			print('Created models dir')
		torch.save(self.state_dict(), './models/{}_epoch_{:02d}'.format(self.name, epoch))


def main():
	batch_size = 64
	epochs = 4
	train_size = 60e3
	iters = int(train_size/batch_size)

	train_loader = DataLoader(
		datasets.MNIST('./datasets/MNIST/', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
			])),
		batch_size=batch_size, shuffle=True)


	cnn = ConvNet()


	for epoch in range(epochs):
		for i in range(iters):
			batch = next(iter(train_loader))
			imgs = batch[0]
			labels = batch[1]

			loss, train = cnn.train(imgs, labels)

			print(loss, train)

if __name__ == '__main__':
	main()
