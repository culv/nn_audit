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

import h5py

from one_layer_pytorch import RandomChoiceShear, SimpleNN
from dim_reduction import PCA, LDA
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

parser.add_argument('--run-dim-red', action='store_true',
    help='run dimensionality reduction on the data')

parser.add_argument('--dim-red-file', type=str, required=True,
    help='file path to save/load PCA model (required)')

parser.add_argument('--p', type=int, default=20,
    help='dimension to project data onto (default=20)')

parser.add_argument('--pca-or-lda', type=str, default='pca',
    help='dimensionality reduction method (lda or pca)')


def prepare_query_point(query, shear):
    """Pass in an image from the test set and a desired shear, and return the
    fully-connected neuron activations of the sheared image"""
    pass


def evaluate_dim_red(test_dataset, train_dataset, knn_model, dim_red_model, p=20):
    """Evaluate the dimensionality reduction method for neural network auditing purposes. For each test
    example, calculate the:
    
        1) percentage of nearest training neighbors that do not have the same class as the 
            test example
        2) variance of nearest train set neighbor shears relative to test example shear

    and average these two measures over the entire test set.

    Args:
        test_dataset = The HDF5 file of neuron activations for the sheared test set
        train_dataset = The HDF5 file for sheared train set
        knn_model = A kNN model fit to the training set neuron activations
        dim_reduce = A valid dimensionality reduction object with a 'project' method (eg. PCA, LDA, etc.)
            that is fit to the training set

    Returns:
        class_err = The average percentage of nearest neighbor training labels different from test
            labels
        shear_var = The average variance of nearest neighbor training shears from test shears
    """

    # Initialize shear_var
    class_err = 0
    shear_var = 0

    # Length of test set
    n_test = test_dataset['shears'].shape[0]

    # Batch size for knn searches
    bs = 512

    # Number of batches
    n_batches = np.ceil(n_test/bs).astype(np.uint)

    # Loop over batches of shears and neuron activations
    b_start = 0
    for b in tqdm(range(n_batches)):
        # Get end index of the batch
        b_end = min(n_test, b_start+bs)

        # Get batch of activations (and project them) and shears and labels
        # Make sure to convert shears to floats since they are stored as 8-bit integers
        a = dim_red_model.project(test_dataset['activations'][b_start:b_end,:], p=p)
        s = np.array(test_dataset['shears'][b_start:b_end]).astype(np.float)
        l = np.array(test_dataset['labels'][b_start:b_end])

        # Get nearest neighbors for neuron batch
        neighbs, _ = knn_model.query(a)

        # Get the shears corresponding to neighbors
        s_neighbs = np.array(train_dataset['shears']).astype(np.float)[neighbs]
        l_neighbs = np.array(train_dataset['labels'])[neighbs]

        # Calculate batch class error and batch variance
        b_err = np.mean(l_neighbs != l[:,np.newaxis])
        b_var = np.mean((s_neighbs-s[:,np.newaxis])**2)

        # Weight batch calculations and then add to totals
        b_weight = (b_end-b_start)/n_test
        class_err += b_weight*b_err
        shear_var += b_weight*b_var

        # Increment batch start to begin from where previous batch ended
        b_start = b_end

    return class_err, shear_var


def main():
    # Get command line args
    args = parser.parse_args()
    print(args)
 
    ### PART 0: Load neuron data, reduce dimensionality, fit/load kNN, load original train/test datasets ###

    # Load train set neurons
    train_neurons = h5py.File('./datasets/neurons/shear_mnist_train_100neurons.hdf5', 'r')
    test_neurons = h5py.File('./datasets/neurons/shear_mnist_test_100neurons.hdf5', 'r')


    if args.pca_or_lda == 'pca':
        # Initialize a PCA model
        dim_red = PCA(args.dim_red_file)

        # Find the PCA eigenvectors and eigenvalues and save
        if args.run_dim_red:
            print('Running PCA...')
            dim_red.run(train_neurons['activations'])
            dim_red.save()
        # Load PCA eigenvectors and eigenvalues
        else:
            print('Loading PCA model...')
            dim_red.load()
    elif args.pca_or_lda == 'lda':
        # Initialize an LDA model
        dim_red = LDA(args.dim_red_file)
        if args.run_dim_red:
            print('Running LDA...')
            dim_red.run(train_neurons['activations'], train_neurons['labels'])
            dim_red.save()
        # Load PCA eigenvectors and eigenvalues
        else:
            print('Loading LDA model...')
            dim_red.load()
    else:
        print('Invalid argument for dimensionality reduction method')
        sys.exit()


    # Project onto first p principal components
    train_neurons_proj = dim_red.project(train_neurons['activations'], p=args.p)

    # Initialize a kNN model
    knn = nnaud.kNN(args.knn_file, k=args.k)

    # Fit a kNN model to the projected data and save
    if args.fit_knn:
        print('Fitting k-nearest neighbors model...')
        knn.fit(train_neurons_proj)
        knn.save()
    # Load kNN model
    else:
        print('Loading k-nearest neighbors model...')
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

    ####### PART 1: Calculate knn shear variance and knn class error ###########################

    print('Evaluating dimensionality reduction method on sheared MNIST...')
    class_err, shear_var = evaluate_dim_red(test_neurons, train_neurons, knn, dim_red, args.p)
    print('class err: ', class_err)
    print('shear var: ', shear_var)

    ####### PART 2: Visualize histograms of randomly sheared test samples ########

    query_shear = 50
    test_set = datasets.MNIST('./datasets/mnist', train=False, download=True,
        transform=transforms.Compose([
            RandomChoiceShear([query_shear]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

    # Get a test sample
    query_image, query_label = test_set[2408] 
    query_image_display = 1.-((query_image*0.3081)+1.307).squeeze()

    # Get neuron activations and prediction
    query_acts, out = fc.forward(query_image.reshape(1,784))
    query_acts = query_acts[0].detach().numpy()
    query_predict = torch.argmax(out, 1)

    # Project query
    query_proj = dim_red.project(query_acts, p=args.p)

    # Get the neighbors in the training set
    neighbs, dist = knn.query(query_proj)

    # Get neighbor shears
    neighbor_shears = np.array([train_neurons['shears'][i] for i in neighbs])

    # Get a second query image with a different shear
    query_shear2 = -10
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
    query_proj2 = dim_red.project(query_acts2, p=args.p)

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