"""
Author: Culver McWhirter
Date:   14 Mar 2018
Time:   15:41
"""

import numpy as np

from torchvision import datasets, transforms

import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import axes3d

from textwrap import wrap

from sklearn import neighbors

import h5py

from dim_reduction import truncated_SVD
from visual_utils import neighbor_acc_visual, show_neighbs_grid

from tqdm import tqdm

import os
import sys

# normalize values to be in [0,1]
# INPUTS: data = data to normalize
# OUTPUTS: normalized data
def normalize(data):
    # normalize data
    d_min = np.min(data)
    d_max = np.max(data)
    return (data - d_min) / (d_max - d_min)
 

# scale and visualize the embedding vectors (supports 2d and 3d plots)
# INPUTS:    dat = the data
#            lab = the labels            
#            title = desired title of the plot
#            plot_type = either plot as 'scatter' or 'digit'
#            samples = whether or not to draw samples of data on plot
# OUTPUTS:   ax = matplotlib.plt Axes obect containing plot of data
def plot_embedding(dat, lab, title=None, plot_type='scatter', samples=False):    
    # get dimension of embedding vectors
    m, dim = dat.shape
    
    if dim == 3:
        fig = plt.figure(dpi=60)
        ax = axes3d.Axes3D(fig) #fig.gca(projection='3d')
        #ax.set_axis_off()

        # loop over all vectors in embedding (same as # of digits)
        if plot_type == 'digit':
            for i in range(m):
                # plot digit as a string, at the location determined by the embedding X
                ax.text(dat[i,0], dat[i,1], dat[i,2], str(lab[i]),
                    color=plt.cm.tab10(lab[i] / 10.), # color determined from Set1 color map
                    fontdict={'weight': 'bold', 'size': 9}) # format font
    
        elif plot_type == 'scatter':
            ax.scatter(dat[:,0], dat[:,1], dat[:,2], c = lab)

    if dim == 2:
        # create 2d figure
        f, ax = plt.subplots()
        ax.scatter(0,0,s=0)
        # loop over all vectors in embedding (same as # of digits)
        if plot_type == 'digit':
            for i in range(m):
                # plot digit as a string, at the location determined by the embedding X
                # NOTE: depending on version of matplotlib, tab10 colormap may be Vega10
                try:
                    ax.text(dat[i,0], dat[i,1], str(lab[i]), # position and string corresponding to digit
                    color=plt.cm.tab10(lab[i] / 10.), # color determined from Set1 color map
                    fontdict={'weight': 'bold', 'size': 9}) # format font
                except:
                    ax.text(dat[i,0], dat[i,1], str(lab[i]),
                            color=plt.cm.Vega10(lab[i] / 10.), # color determined from Set1 color map
                            fontdict={'weight': 'bold', 'size': 9}) # format font
        elif plot_type == 'scatter':
            ax.scatter(dat[:,0], dat[:,1], c=lab)
        if samples:
            # show digit images on plot
            if hasattr(offsetbox,  'AnnotationBbox'): # will only work with matplotlib versions past v1.0
                # initialize shown images locations array with upper right corner of plot
                shown_images = np.array([[1.,1.]])
                # loop over all digits
                for i in range(m):
                    # calculate squared distance between current image's embedding vector and all others that have been displayed
                    dist = np.sum((dat[i] - shown_images) ** 2, 1)
                    # if the smallest squared distance is below threshold, don't print it (to ensure plot isn't overcrowded)
                    if np.min(dist) < 4e-3:
                        continue
                    # otherwise, add embedding vector to array of shown images
                    shown_images = np.r_[shown_images, [dat[i]]]
                    # and display image of digit at the embedding vector location
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(digits.images[i],  cmap=plt.cm.gray_r),
                        dat[i])
                    ax.add_artist(imagebox)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if dim == 3:
        ax.set_zlim(0,1)
    if title is not None:
        ax.set_title(title)
    return fig, ax


# check out what the nearest neighbors look like
# INPUTS:   test_image = np.array of test image
#           knn_image_list = Python list of k-nearest neighbor images in each embedding
#               i.e. [[images for dim1], [images for dim2],...]
#           dim_list = list of dimensions of embeddings
# OUTPUTS:  fig = matplotlib figure containing the images
#           ax = array of matplotlib Axes objects containing the images
def visualize_neighbors(test_image, knn_accuracy, knn_image_list, dim_list):
    # number of subplots to make = number of dims + original test image
    n_plots = len(dim_list)+1
    # number of neighbors
    k = knn_image_list[0].shape[0]
    size = np.sqrt(knn_image_list[0].shape[1]).astype('int')
    # make figure and Axes objects for subplots
    fig, ax = plt.subplots(nrows=np.ceil(n_plots/2.).astype('int'), ncols=2)
    # set figure size
    fig.set_size_inches(8, 11.5)
    
    # axis array from 2D->1D to make it easier to loop over
    ax = ax.flatten() 
    
    # format test and nearest neighbor images into grid
    n_img_per_row = np.sqrt(k).astype('int')
    test_img = np.zeros(((size+2)*n_img_per_row, (size+2)*n_img_per_row))
    test_img[30*2+1:30*2+28+1, 30*2+1:30*2+28+1] = 1-np.reshape(test_image, (28,28))

    grid_list = []
    for image_list in knn_image_list:
        neighb_img = np.ones((30*n_img_per_row, 30*n_img_per_row))
        for i in range(n_img_per_row):
            ix = 30 * i
            for j in range(n_img_per_row):
                iy = 30*j
                neighb_img[ix+1:ix+29, iy+1:iy+29] = 1-image_list[i*n_img_per_row+j].reshape((28,28))
        np.pad(neighb_img, [2,2], 'constant', constant_values=1)
        grid_list.append(neighb_img)

    # 1st subplot will be test image
    im = ax[0].imshow(test_img, cmap=plt.cm.binary)
    ax[0].set_title('New test set image')

    # 2nd subplot will be accuracy of 250 k-nearest neighbors
    ax[1].bar(dim_list, knn_accuracy, color='g')
    ax[1].set_title('kNN accuracy by dimension for k={} neighbors'.format(250))

    # rest of subplots will be k-nearest neighbors for each dimension
    for i, dim in enumerate(dim_list):
        im = ax[i+2].imshow(grid_list[i], cmap=plt.cm.binary)
        title = '{} nearest training set neighbors in {}-dim neuron-space'.format(k, dim)
        wrapped_title = '\n'.join(wrap(title, 40))
        ax[i+2].set_title(wrapped_title)

    # remove borders and axis marks
    fig.patch.set_visible(False)
    for this_ax in ax:
        this_ax.patch.set_visible(False) # remove borders
        this_ax.axis('off') # remove axes

    # put axes and borders back on accuracy subplot and set axes limits
    ax[1].axis('on')
    ax[1].set_xlim(0,7)
    ax[1].set_ylim(0,1)
    ax[1].grid('on')
#    ax[1].patch.set_visible(True)
    
    return fig, ax


def find_training_neighbors(test_point, training_points, training_images, training_labels, k):
    """Given a test point, find the nearest neighbor images and corresponding labels in the training set

    Args:
        test_point = The query point
        training_points = A NumPy array of all the training points (in the reduced dimension space)
        k = Number of neighbors to find
        training_images = A NumPy array of all the raw training images
        training_labels = A NumPy array of all the training labels

    Returns:
        nearest_images = The top k nearest images
        nearest_labels = The top k nearest labels
    """

    # If training points are 1-dim, add axes to make it a 2D array            
    if test_point.shape[-1] == 1:
        training_points = training_points.reshape(1,-1).T

    # Fit kNN to the training data
    knn = neighbors.NearestNeighbors(n_neighbors = k)
    knn.fit(training_points)

    # Get the distances and indices of top k nearest neighbors in the train set
    dist, neighbs = knn.kneighbors(test_point[np.newaxis,:])

    # Get rid of singleton dimension
    neighbs = np.squeeze(neighbs)

    # Get nearest images and their labels
    nearest_images = training_images[neighbs]
    nearest_labels = training_labels[neighbs]

    return nearest_images, nearest_labels


def main():
    # Load neuron data for test and train sets
    train_neurons = h5py.File('./datasets/neurons/compressed_mnist_train_100neurons.hdf5', 'r')
    test_neurons = h5py.File('./datasets/neurons/compressed_mnist_test_100neurons.hdf5', 'r')

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

    """
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
    """

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


if __name__ == '__main__':
    main()