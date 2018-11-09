"""
Author: Culver McWhirter
Date:   7 Nov 2018
Time:   16:25
"""

import numpy as np

import torch
from torchvision import datasets, transforms

import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault
import matplotlib.pyplot as plt

import h5py

from audit.dim_reduction import PCA
from audit.visual_utils import show_neighbs_grid2, plot_embedding
from audit.nn_audit_utils import kNN

import os
import sys


def main():
    # Load neuron data for test and train sets
    train_neurons = h5py.File('./datasets/neurons/conv_mnist_train.hdf5', 'r')
    test_neurons = h5py.File('./datasets/neurons/conv_mnist_test.hdf5', 'r')

    # Fit PCA to the data
    pca = PCA('./models/conv_mnist_PCA.hdf5')
    if False:
        pca.run(train_neurons['activations'])
        pca.save()
    else:
        pca.load()

    # Get explained variance PCA as a function of number of principal components
    explained_var = np.cumsum(pca.D)/np.sum(pca.D)

    # Get number of principal components that preserve 95% of variance
    n_pca = np.where(explained_var>=0.90)[0][0]

    # Fit kNN to the PCA'ed data
    knn = kNN('./models/conv_mnist_PCA_kNN.joblib')
    if False:
        knn.fit(pca.project(train_neurons['activations'], n_pca))
        knn.save()
    else:
        knn.load()

    # Plot explained variance of PCA
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 513, 1), explained_var, c='b', linewidth=2)
    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Variance ratio')
    ax.set_title('Variance explained by PCA')
    ax.set_xlim(0,512)
    ax.set_ylim(0,1.05)
    ax.grid()
    ax.set_xticks(np.arange(0,513,10), minor=True)
    ax.grid(which='minor')
    ax.set_yticks(np.arange(0,1.01,.1))
    plt.show()

    # Pick a query point from the test set
    q_id = 274

    # Find the k nearest images and their labels in the training set
    nearest_idcs, dist = knn.query(pca.project(test_neurons['activations'][q_id], n_pca))

    # Get the corresponding images
    # Get the MNIST training set
    train_set = datasets.MNIST('./datasets/mnist', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    # Get the MNIST test set
    test_set = datasets.MNIST('./datasets/mnist', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    nearest_images = train_set.train_data[nearest_idcs]
    query_image = test_set.test_data[q_id]

    # Visualize
    fig2, ax2 = plt.subplots()
    ax2 = show_neighbs_grid2(ax2, query_image, nearest_images, grid_size=4)
    plt.show()

    # Show the PCA projection of the data
    fig3, ax3 = plot_embedding(pca.project(train_neurons['activations'], 2), train_neurons['labels'], 
        title='Data projected onto top 3 principal components')
    plt.show()


    # # Visualize accuracy & variance captured by PCA as a function of dimension
    # accs = []

    # dims = np.linspace(1,9,9)
    # for i in tqdm(range(9)):
    #     # Reduce the dimension of train and test sets
    #     train_neurons_embed = np.matmul(train_neurons['activations'], evecs[:, 0:i+1])
    #     test_neurons_embed = np.matmul(test_neurons['activations'], evecs[:, 0:i+1])
            
    #     # Find the k nearest images and their labels in the training set
    #     nearest_images, nearest_labels = find_training_neighbors(test_neurons_embed[query_idx], 
    #         train_neurons_embed, train_images, train_labels, k)

    #     # Get accuracy and append
    #     accs.append(np.mean(nearest_labels==query_label))

    # # Convert accuracy to NumPy array
    # accs = np.array(accs)

    # fig, ax = plt.subplots()
    # ax = neighbor_acc_visual(ax, accs, cum_var_ratio[:9], dims, query_image)
    # plt.show()
    
if __name__ == '__main__':
    main()