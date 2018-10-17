"""
Author: Culver McWhirter
Date:   16 Oct 2018
Time:   18:53
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
from dim_reduction import PCA
from visual_utils import shear, neighbor_acc_visual, show_neighbs_grid, plot_embedding
import nn_audit_utils as nnaud

from tqdm import tqdm

import os
import sys

import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='Neural network audit experiment on sheared MNIST data')

parser.add_argument('--fit-knn', action='store_true',
    help='fit the kNN model and save it')

parser.add_argument('--k', type=int, default=500,
    help='number of nearest neighbors for kNN model (default=500)')

parser.add_argument('--knn-file', type=str, required=True,
    help='file path to save/load kNN model (required)')

parser.add_argument('--run-pca', action='store_true',
    help='run PCA on the data')

parser.add_argument('--pca-file', type=str, required=True,
    help='file path to save/load PCA model (required)')

parser.add_argument('--p', type=int, default=20,
    help='number of principal components to project onto (default=20)')


def prepare_query_point(query, shear):
    """ """
    pass


def main():
    # Get command line args
    args = parser.parse_args()
    print(args)
 
    ### PART 0: Load neuron data, reduce dimensionality, fit/load kNN, load original train/test datasets ###

    # Load train set neurons
    train_neurons = h5py.File('./datasets/neurons/shear_mnist_train_100neurons.hdf5', 'r')
    test_neurons = h5py.File('./datasets/neurons/shear_mnist_test_100neurons.hdf5', 'r')


    # Initialize a PCA model
    pca = PCA(args.pca_file, args.p)

    # Find the PCA eigenvectors and eigenvalues and save
    if args.run_pca:
        pca.run(train_neurons['activations'])
        pca.save()
    # Load PCA eigenvectors and eigenvalues
    else:
        pca.load()

    # Project onto first p principal components
    train_neurons_proj = pca.project(train_neurons['activations'])

    # Initialize a kNN model
    knn = nnaud.kNN(args.knn_file)

    # Fit a kNN model to the projected data and save
    if args.fit_knn:
        knn.fit(train_neurons_proj)
        knn.save()
    # Load kNN model
    else:
        knn.load()

    # Load MNIST train and test sets
    train_set = datasets.MNIST('./datasets/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

    # Load network
    fc = SimpleNN()
    fc.load_state_dict(torch.load('./models/shear_fc100_09.pt'))


    ####### PART 1: Visualize histograms of randomly sheared test samples ########

    query_shear = 45.72
    test_set = datasets.MNIST('./datasets/mnist', train=False, download=True,
        transform=transforms.Compose([
            RandomChoiceShear([query_shear]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))    

    # Get a test sample
    query_image, query_label = test_set[2406] 
    query_image_display = 1.-((query_image*0.3081)+1.307).squeeze()

    # Get neuron activations and prediction
    query_acts, out = fc.forward(query_image.reshape(1,784))
    query_acts = query_acts[0].detach().numpy()
    query_predict = torch.argmax(out, 1)

    # Project query
    query_proj = pca.project(query_acts)

    # Get the neighbors in the training set
    neighbs, dist = knn.query(query_proj)

    # Get neighbor shears
    neighbor_shears = np.array([train_neurons['shears'][i] for i in neighbs])

    # Get a second query image with a different shear
    query_shear2 = -17.34
    test_set2 = datasets.MNIST('./datasets/mnist', train=False, download=True,
        transform=transforms.Compose([
            RandomChoiceShear([query_shear2]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

    # Get a test sample
    query_image2, query_label2 = test_set2[999] 
    query_image_display2 = 1.-((query_image2*0.3081)+1.307).squeeze()

    # Get neuron activations and prediction
    query_acts2, out2 = fc.forward(query_image2.reshape(1,784))
    query_acts2 = query_acts2[0].detach().numpy()
    query_predict2 = torch.argmax(out2, 1)

    # Project and find nearest neighbors
    query_proj2 = pca.project(query_acts2)

    # Get the distances and indices of top k nearest neighbors in the train set
    neighbs2, dist2 = knn.query(query_proj2)    

    # Get neighbor shears
    neighbor_shears2 = np.array([train_neurons['shears'][i] for i in neighbs2])

    fig, axs = plt.subplots(2,2)

    axs[0][0].imshow(query_image_display, cmap=plt.cm.gray_r)
    axs[0][0].set_title('Test image with a shear of {:.2f}$\degree$'.format(query_shear))
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])

    axs[0][1].hist(neighbor_shears, bins=np.linspace(-55,55,12), edgecolor='black')
    axs[0][1].axvline(x=query_shear, c='r')
    axs[0][1].set_title('Histogram of shears for k={:d} nearest train-set neighbors'.format(args.k))
    axs[0][1].set(xticks=np.linspace(-50,50,11), xlim=[-60,60])
    axs[0][1].set_xlabel('shear angle')
    axs[0][1].set_ylabel('count')
    axs[0][1].grid()

    axs[1][0].imshow(query_image_display2, cmap=plt.cm.gray_r)
    axs[1][0].set_title('Test image with a shear of {:.2f}$\degree$'.format(query_shear2))
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])

    axs[1][1].hist(neighbor_shears2, bins=np.linspace(-55,55,12), edgecolor='black')
    axs[1][1].axvline(x=query_shear2, c='r')
#    axs[1][1].set_title('Histogram of shears for k={:d} nearest train-set neighbors'.format(args.k))
    axs[1][1].set(xticks=np.linspace(-50,50,11), xlim=[-60,60])
    axs[1][1].set_xlabel('shear angle')
    axs[1][1].set_ylabel('count')
    axs[1][1].grid()

    plt.show()

    ######## PART 2: Calculate metrics ########


if __name__ == '__main__':
	main()