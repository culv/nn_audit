"""
Author: Culver McWhirter
Date:   14 Mar 2018
Time:   15:41
"""

import numpy as np

import torch
from torchvision import datasets, transforms

import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault
import matplotlib.pyplot as plt

from sklearn import neighbors

import h5py

from one_layer_pytorch import RandomChoiceShear, SimpleNN
from dim_reduction import truncated_SVD
from visual_utils import shear, neighbor_acc_visual, show_neighbs_grid, plot_embedding
from nn_audit_utils import find_training_neighbors

from tqdm import tqdm

import os
import sys

def main():
    # Basic experiment
    
    # Load neuron data for test and train sets
    train_neurons = h5py.File('./datasets/neurons/mnist_train_100neurons.hdf5', 'r')
    test_neurons = h5py.File('./datasets/neurons/mnist_test_100neurons.hdf5', 'r')

    # Do SVD to find projection matrix
    evecs, evals = truncated_SVD(train_neurons['activations'])

    # Get the MNIST training set
    train_set = datasets.MNIST('./datasets/mnist', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    # Get labels and images (as 784-d vectors)
    train_labels = train_set.train_labels.numpy()
    train_images = train_set.train_data.numpy()/255.

    # Get the MNIST test set
    test_set = datasets.MNIST('./datasets/mnist', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    # Get labels and images (as 784-d vectors)
    test_labels = test_set.test_labels.numpy()
    test_images = test_set.test_data.numpy()/255.

    # Get percentage of variance captured by each dimension of PCA
    cum_var_ratio = np.cumsum(evals)/np.sum(evals)

    # Number of neighbors
    k = 500

    # Indices of test images to check
    query_idx = 12

    # The query image and label
    query_image = test_images[query_idx]
    query_label = test_labels[query_idx]

    # Visualize accuracy & variance captured by PCA as a function of dimension
    accs = []

    dims = np.linspace(1,9,9)
    for i in tqdm(range(9)):
        # Reduce the dimension of train and test sets
        train_neurons_embed = np.matmul(train_neurons['activations'], evecs[:, 0:i+1])
        test_neurons_embed = np.matmul(test_neurons['activations'], evecs[:, 0:i+1])
            
        # Find the k nearest images and their labels in the training set
        nearest_images, nearest_labels = find_training_neighbors(test_neurons_embed[query_idx], 
            train_neurons_embed, train_images, train_labels, k)

        # Get accuracy and append
        accs.append(np.mean(nearest_labels==query_label))

    # Convert accuracy to NumPy array
    accs = np.array(accs)

    fig, ax = plt.subplots()
    ax = neighbor_acc_visual(ax, accs, cum_var_ratio[:9], dims, query_image)
    plt.show()

    # Visualize the nearest images in the training set
    # Dimensions to project into
    dim = 5

    # Reduce the dimension of train and test sets
    train_neurons_embed = np.matmul(train_neurons['activations'], evecs[:, 0:dim])
    test_neurons_embed = np.matmul(test_neurons['activations'], evecs[:, 0:dim])

    # Find the k nearest images and their labels in the training set
    nearest_images, nearest_labels = find_training_neighbors(test_neurons_embed[query_idx], 
        train_neurons_embed, train_images, train_labels, k)

    fig2, ax2 = plt.subplots()

    ax2 = show_neighbs_grid(ax2, query_image, nearest_images, dim)

    plt.show()

    # Show the data in 2-d or 3-d
    # Reduce the dimension of train set
    train_neurons_embed = np.matmul(train_neurons['activations'], evecs[:, 0:2])

    fig, ax = plot_embedding(train_neurons_embed, train_labels, train_images, title='Test title')
    plt.show()
    
if __name__ == '__main__':
    main()