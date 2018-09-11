"""
Author: Culver McWhirter
Date:	11 Sep 2018
Time:	13:48
"""

import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import numpy as np

import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import axes3d

import sys
import os

def neighbor_acc_visual(ax, accuracies, captured_var, dims, image):
    """Visualize the accuracy of the training set nearest neighbors (i.e. neighbors that have the same label as
    the query image) as a function of the dimension the neuron activations are reduced to. The variance captured
    by PCA as a function of dimension is also shown for comparison. The query image is also shown in the top left
    for reference

    Args:
        ax = A matplotlib Axes object to plot on
        accuracies = A NumPy array of accuracies in the range [0,1]
        captured_var = A NumPy array of the variance captured by PCA for each dimension
        dims = A NumPy array of the relevant dimensions
        image = The query image

    Returns:
        ax = The finished Axes object
    """

    # Draw scatter plots for the kNN accuracy and the captured variance vs. dimension
    acc = ax.scatter(dims, 100*accuracies, c='g', s=7)
    var = ax.scatter(dims, 100*captured_var, c='b', s=5)

    # Labels, axes limits, tick marks, etc.
    ax.set_xlabel('dimension')
    ax.set_ylabel('% Variance Preserved/% Matching Neighbors')
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.grid()
    ax.set_xticks(np.arange(0,101,1), minor=True)
    ax.set_yticks(np.arange(0,101,10))
    ax.grid(which='minor')
    ax.legend((acc, var), ('% kNN w/ same label', '% variance preserved by PCA'))

    # Show query image    
    imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(1.-image, cmap=plt.cm.gray_r),
            (5,95), pad=0)
    ax.add_artist(imagebox)
        
    return ax
   

def show_neighbs_grid(ax, query_image, neighbor_images, dim, grid_size=6):
    """Show a grid of nearest neighbors in the training set. Query image is shown in the top left,
    and the neighbors are shown in the rest of the grid. Grid has 'grid_edge' number of images along
    each side

    Args:
        ax = A matplotlib Axes object to plot on
        query_image = The test image
        neighbor_images = The nearest training images
        grid_edge = The number of images along an edge

    Returns:
        ax = The finished Axes object
    """

    # Grab number of nearest neighbors needed for the grid
    neighbor_images = neighbor_images[:grid_size**2-1, :, :]

    # Concatenate query image with neighbor images
    images = np.concatenate((query_image[np.newaxis,:,:], neighbor_images), 0)

    # Image size in pixels
    im_size = neighbor_images.shape[-1]

    # Create blank grid
    grid_size_in_pixels = grid_size*im_size + (grid_size+2)*2
    grid = np.ones([grid_size_in_pixels, grid_size_in_pixels])

    # Loop over image slots
    for j in range(grid_size):
        # The y-coordinate
        jx = j*(im_size+2)+2
        for k in range(grid_size):
            # The x-coordinate
            kx = k*(im_size+2)+2
            grid[kx:kx+im_size, jx:jx+im_size] = 1.-images[k+j*grid_size]

    # Add border around query image and the whole grid
    g = 0.5
    grid[im_size:im_size+4, 0:im_size+4] = g
    grid[0:im_size+4, im_size:im_size+4] = g
    grid = np.pad(grid, ((4,4),(4,4)), 'constant', constant_values=((g,g),(g,g)))        

    # Place grid image inside the Axes object
    ax.imshow(grid, cmap=plt.cm.gray_r)

    # Remove axes and borders
    ax.axis('off')
    ax.patch.set_visible('off')

    # Set title
    ax.set_title('{} Nearest Test Set Images in {}-dim Neuron Space'.format(grid_size**2-1, dim))

    return ax


def shear(image, shear):
	"""Take a PyTorch Tensor object, shear it using 'shear' as the angle, then return
	the sheared image (also as a Tensor)"""

	xfm_to_PIL = transforms.ToPILImage()
	xfm_to_Tensor = transforms.ToTensor()
	sheared_image = xfm_to_Tensor( transforms.functional.affine( xfm_to_PIL(image), 
		angle=0, translate=(0,0), scale=1, shear=shear) )

	return sheared_image

def visualize_shear(axs, image, shear_list, labels=True):
	"""Visualize the progression of shearing an image for several different shearing
	angles in a list

	Args:
		axs = List of matplotlib Axes objects
		image = The original image
		shear_list = The list of shearing angles
		labels = Whether or not to write the shearing angle over each image

	Returns:
		axs = List of finished plots
	"""

	for i, shear_angle in enumerate(shear_list):
		axs[i].imshow(1.-shear(image, shear_angle).squeeze(), cmap=plt.cm.binary)
		if labels:
			axs[i].set_title('{}$^\circ$'.format(shear_angle))
		axs[i].axis('off')

	return axs

def main():
	# Load MNIST training set
	train_set = datasets.MNIST('./datasets/mnist', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				]))

	im = train_set[0][0]

	shears = [-45, -30, -15, 0, 15, 30, 45]

	fig, axs = plt.subplots(5, len(shears))

	axs[0] = visualize_shear(axs[0], train_set[0][0], shears)
	axs[1] = visualize_shear(axs[1], train_set[33][0], shears, labels=False)
	axs[2] = visualize_shear(axs[2], train_set[146][0], shears, labels=False)
	axs[3] = visualize_shear(axs[3], train_set[24440][0], shears, labels=False)
	axs[4] = visualize_shear(axs[4], train_set[50001][0], shears, labels=False)

	fig.suptitle('MNIST images with varying shear angle')

	plt.show()

if __name__ == '__main__':
	main()