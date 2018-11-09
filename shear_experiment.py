"""
Author: Culver McWhirter
Date:   9 Nov 2018
Time:   13:10
"""

import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault
import matplotlib.pyplot as plt

import h5py

from networks.mnist_convnet import RandomChoiceShear, ConvNet
from audit.dim_reduction import PCA, LDA
from audit.visual_utils import shear, show_neighbs_grid2, plot_embedding, get_Axes_size, set_Axes_size
from audit.nn_audit_utils import kNN

from tqdm import tqdm

import os
import sys

# Load MNIST train and test sets as global variables
mnist_train = datasets.MNIST('./datasets/mnist', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        ]))

mnist_test = datasets.MNIST('./datasets/mnist', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        ]))

# Load ConvNet, also global
cnn = ConvNet()
cnn.load_state_dict(torch.load('./networks/shear_conv_mnist_09.pt'))
cnn.eval()



def get_query(q_id, angle):
    """Pass index of an image from the test set and a desired shear, and return the
    fully-connected neuron activations of the sheared image"""

    # Grab image (note that you'll need to add a singleton dimension for color channel)
    q = mnist_test.test_data[q_id][None, :, :]

    # Convert to PIL, shear, convert back to Tensor, and normalize with mean=0.1307, std=0.3081
    q = (shear(q, angle) - 0.1307)/0.3081

    # Get fully-connected neuron activations (note you need to add a singleton dimension again for batch size)
    q, _ = cnn(q[None, :, :, :])

    return q.detach().numpy()

def shear_histogram(ax, shear, data, title=None):
    """Pass a Matplotlib Axis object, shear (float), data (Python array) and a title (optional) to construct
    a histogram of relative frequencies of shears"""

    # Shear bins
    bins = np.linspace(-50, 50, 11)
    # Get relative frequencies
    rel_f = np.array([sum(data==s) for s in bins])/float(len(data))
    # Make histogram
    ax.bar(bins, rel_f, width=10, edgecolor='black')
    # Vertical line where specified shear is
    ax.axvline(x=shear, c='r')
    # Set title, axes labels, etc.
    ax.set_title(title)
    ax.set(xticks=np.linspace(-50,50,11), xlim=[-55,55])
    ax.set_xlabel('Shear angle')
    ax.set_ylabel('Relative frequency')
    ax.grid()
    ax.set_axisbelow(True)

    return ax

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
    bs = 256

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
    # Get query activations
    q_id = 779
    q_shear = 25.3
    q = get_query(q_id, q_shear)

    # Get query image
    q_image = 255*shear(mnist_test.test_data[q_id][None,:,:], q_shear).squeeze().numpy()

    # Load neurons
    train_neurons = h5py.File('./datasets/neurons/conv_shear_mnist_train.hdf5', 'r')
    test_neurons = h5py.File('./datasets/neurons/conv_shear_mnist_test.hdf5', 'r')

    # Do PCA
    pca = PCA('./models/conv_shear_PCA.hdf5')
    if False:
        pca.run(test_neurons['activations'])
        pca.save()
    else:
        pca.load()

    # Get explained variance PCA as a function of number of principal components
    explained_var = np.cumsum(pca.D)/np.sum(pca.D)

    # Get number of principal components that preserve 95% of variance
    n_pca = np.where(explained_var>=0.90)[0][0]

    # Fit kNN to the PCA'ed data
    knn = kNN('./models/conv_shear_PCA_kNN.joblib')
    if False:
        knn.fit(pca.project(train_neurons['activations'], n_pca))
        knn.save()
    else:
        knn.load()

    # Find nearest neighbors to query
    neighbor_ids, dist = knn.query(pca.project(q, n_pca))

    # Get shears of neighbors
    neighbor_shears = [train_neurons['shears'][i] for i in neighbor_ids]

    # Get images of neighbors
    neighbor_images = [mnist_train.train_data[i%len(mnist_train)] for i in neighbor_ids]
    neighbor_images = [shear(neighbor_images[i][None,:,:], neighbor_shears[i]).numpy() for i in range(len(neighbor_images))]
    neighbor_images = 255*np.concatenate(neighbor_images, 0)

    # Plot the query image, nearest training images, and histogram of shears
    fig, ax = plt.subplots()

    ax = show_neighbs_grid2(ax, q_image, neighbor_images, grid_size=4, buff=10)
    ax.set_title('Query image ({:.1f}$\degree$ shear) and nearest training neighbors'.format(q_shear))

    fig2, ax2 = plt.subplots()
    ax2 = shear_histogram(ax2, q_shear, neighbor_shears, title='Frequency of shears of 500 nearest training neighbors')

    plt.show()

    # Evaluate PCA kNN class and shear accuracy
    class_err, shear_var = evaluate_dim_red(test_neurons, train_neurons, knn, pca, n_pca)
    print('class err: {:.2f}'.format(100*class_err))
    print('shear std: {:.2f} degrees'.format(shear_var))

if __name__ == '__main__':
	main()