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

import seaborn as sns

import h5py

from audit.dim_reduction import PCA, LDA
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


    # Fit LDA to the data
    """
    lda = LDA('./models/conv_mnist_LDA.hdf5')
    if False:
        lda.run(train_neurons['activations'], train_neurons['predictions'])
        lda.save()
    else:
        lda.load()

    # Fit kNN to the LDA'ed data
    knn = kNN('./models/conv_mnist_LDA_kNN.joblib')
    if False:
        knn.fit(lda.project(train_neurons['activations']))
        knn.save()
    else:
        knn.load()
    """

    # Plot explained variance of PCA

    sns.set()
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 513, 1), explained_var, c='b', linewidth=2)
    ax.axvline(x=n_pca, c='r')
    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Variance ratio')
    ax.set_title('Variance explained by PCA')
#    ax.grid()
    ax.set_xticks(np.arange(0,513,10), minor=True)
    ax.set_xticks(np.concatenate((ax.get_xticks(), [n_pca]), 0))
#    sys.exit()
    ax.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax.grid(which='minor')
    ax.set_yticks(np.arange(0,1.01,.1))
    ax.set_xlim(0,512)
    ax.set_ylim(0,1.05)
#    plt.savefig('./figures/mnist_convnet_pca_variance.png', bbox_inches='tight')


    # Indices of misclassified
    test_labels = np.array(test_neurons['labels'])
    test_predicts = np.array(test_neurons['predictions'])
    wrong = np.where(test_labels!=test_predicts)[0]
    print(wrong)

    # Pick a query point from the test set
    # Normal 9, correct:         274
    # Weird 4, correct:          8000
    # S-shaped 5, correct:       333
    # 7 with bar, wrong as 2:    9009
    # 8 open top, wrong as 4:    4137
    q_id = 8000

    # Find the k nearest images and their labels in the training set
    nearest_idcs, dist = knn.query(pca.project(test_neurons['activations'][q_id], n_pca))
#    nearest_idcs, dist = knn.query(lda.project(test_neurons['activations'][q_id]))

    # Get the corresponding images
    # Get the MNIST training set
    script_dir = os.path.dirname(os.path.realpath(__file__))
    mnist_dir = os.path.join(script_dir, 'datasets', 'mnist')

    train_set = datasets.MNIST(mnist_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    # Get the MNIST test set
    test_set = datasets.MNIST(mnist_dir, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    nearest_images = train_set.train_data[nearest_idcs]
    query_image = test_set.test_data[q_id]

    print('Query label:\t', test_neurons['labels'][q_id])
    print('Query predict:\t', test_neurons['predictions'][q_id])
    print('Neighbor labels:\t', [train_neurons['labels'][i] for i in nearest_idcs[0:12]])
    print('Neighbor predicts:\t', [train_neurons['predictions'][i] for i in nearest_idcs[0:12]])
    
    # Visualize
    fig2, ax2 = plt.subplots()
    show_neighbs_grid2(ax2, query_image, nearest_images, grid_size=(4, 3))
    ax2.set_title('    Query                             Neighbors', loc='left')

#    plt.savefig('./figures/mnist_audit_8_wrong_as_4.png', bbox_inches='tight')

 
    # Show the PCA projection of the data
    sns.set()
    fig3, ax3 = plot_embedding(pca.project(train_neurons['activations'], 3), train_neurons['labels'], 
        title='Data projected onto top 3 principal components')
    plt.show()

    
if __name__ == '__main__':
    main()