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


def get_Axes_size(ax):
    """Get the size of a Matplotlib Axes object (in pixels)"""
    bbox = ax.get_window_extent()
    return bbox.height, bbox.width

def set_Axes_size(ax, h, w, dpi=80):
    """Set the size of a Matplotlib Axes object (in pixels)"""
    ax.figure.set_size_inches(h/dpi, w/dpi)


def normalize(data):
    """Normalize data to be in [0,1]"""
    d_min = np.min(data)
    d_max = np.max(data)
    return (data - d_min) / (d_max - d_min)

def plot_embedding(data, labels, title=None, plot_type='scatter', images=None):    
    """Visualize the data as a scatter plot in a reduced dimension (either 2-d or 3-d). You can also
    optionally display sample images to give an idea of the data subspace, and display each data point by
    its class label (integers 0-9) instead of a point

    Args:
        ax = A matplotlib Axes object to plot in
        data = The data to be visualized
        labels = Class labels corresponding to data
        images = The original images corresponding to data
        title = Title of the plot
        point_type = Plot points either as points in a scatter plot, or as digits
        samples = Whether or not to display sample images on the plot

    Returns:
        fig, ax = The finished plot (Figure and Axes objects)
    """

    # Get colormap (depending on version of Matplotlib it will be 'tab10' or 'Vega10')
    try:
        cm = plt.cm.tab10
    except:
        cm = plt.cm.Vega10

    # Get the dimensions of the data
    m, dim = data.shape

    # Normalize the data
    data = normalize(data)    

    # Plot for 3-d data    
    if dim == 3:
        fig = plt.figure(dpi=60)
        ax = axes3d.Axes3D(fig) #fig.gca(projection='3d')
        #ax.set_axis_off()

        # Loop over all data vectors, plotting either as points or digits
        if plot_type == 'digit':
            for i in range(m):
                # Plot digit as a string, at the location determined by the data point
                # Color is determined from colormap
                ax.text(data[i,0], data[i,1], data[i,2], str(labels[i]),
                    color=cm(labels[i]),
                    fontdict={'weight': 'bold', 'size': 9})

        elif plot_type == 'scatter':
            ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap=cm)

    # Plot for 2-d data
    if dim == 2:
        fig, ax = plt.subplots()
        ax.scatter(0,0,s=0)

        # Loop over all data vectors, plotting either as points or digits
        if plot_type == 'digit':
            for i in range(m):
                # Plot digit as a string, at the location determined by the data point
                # Color is determined from colormap
                ax.text(data[i,0], data[i,1], data[i,2], str(labels[i]),
                    color=cm(labels[i]),
                    fontdict={'weight': 'bold', 'size': 9})

        elif plot_type == 'scatter':
            ax.scatter(data[:,0], data[:,1], c=labels, cmap=cm)

        # Show sample images if desired (will only work with matplotlib v1.0+)
        if (images != None) and hasattr(offsetbox,  'AnnotationBbox'):

            # Initialize shown images locations array, starting with upper right corner of plot
            shown_images = np.array([[1.,1.]])

            # Loop over all data points
            for i in range(m):

                # Calculate squared distance between current image's data point and all others that have already been displayed
                dist = np.sum((dat[i] - shown_images) ** 2, 1)

                # If the smallest squared distance is below threshold, don't display (this ensures plot isn't overcrowded)
                if np.min(dist) < 4e-3:
                    continue

                # Otherwise, add data point to array of shown images and display the image at the corresponding location
                shown_images = np.r_[shown_images, [data[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i],  cmap=plt.cm.gray_r),
                    data[i])

                ax.add_artist(imagebox)

    # Set axes limits, title, etc.
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if dim == 3:
        ax.set_zlim(0,1)
    if title is not None:
        ax.set_title(title)

    return fig, ax

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

def show_neighbs_grid(ax, query_image, neighbor_images, grid_size=6):
    """Show a grid of nearest neighbors in the training set. Query image is shown in the top left,
    and the neighbors are shown in the rest of the grid. Grid has 'grid_size' number of images along
    each side

    Args:
        ax = A matplotlib Axes object to plot on
        query_image = The test image
        neighbor_images = The nearest training images
        grid_size = The number of images along an edge

    Returns:
        ax = The finished Axes object
    """

    # Grab number of nearest neighbors needed for the grid
    neighbor_images = neighbor_images[:grid_size**2-1, :, :]

    # Concatenate query image with neighbor images
    images = np.concatenate((query_image[None,:,:], neighbor_images), 0)

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
            grid[kx:kx+im_size, jx:jx+im_size] = images[k+j*grid_size]

    # Add border around query image and the whole grid
    g = 0.7*255
    grid[im_size+1:im_size+2, 0:im_size+1] = g
    grid[0:im_size+2, im_size+1:im_size+2] = g
    grid = np.pad(grid, ((1,1),(1,1)), 'constant', constant_values=((g,g),(g,g)))        

    # Place grid image inside the Axes object
    ax.imshow(grid, cmap=plt.cm.gray)

    # Remove axes and borders
    ax.axis('off')
    ax.patch.set_visible('off')

    return ax

def show_neighbs_grid2(ax, query_image, neighbor_images, grid_size=6, buff=10):
    """Show a grid of nearest neighbors in the training set. Query image is shown in the top left,
    and the neighbors are shown in a separate grid. Grid has 'grid_size' number of images along
    each side

    Args:
        ax = A matplotlib Axes object to plot on
        query_image = The test image
        neighbor_images = The nearest training images
        grid_size = The number of images along an edge
        buff = Number of pixels in between query image and neighbor images

    Returns:
        ax = The finished Axes object
    """

    # Grab number of nearest neighbors needed for the grid
    neighbor_images = neighbor_images[:grid_size**2, :, :]

    # Image size in pixels
    im_size = neighbor_images.shape[-1]

    # Create blank grid, accounting for 1-pixel separation between images
    grid_size_in_pixels = grid_size*im_size + (grid_size+1)
    grid = 255*np.ones([grid_size_in_pixels, grid_size_in_pixels])

    # Loop over neighbor images and add them to the grid
    for j in range(grid_size):
        # The x-coordinate
        x = j*(im_size+1)+1
        for k in range(grid_size):
            # The y-coordinate
            y = k*(im_size+1)+1
            grid[y:y+im_size, x:x+im_size] = neighbor_images[k+j*grid_size]

    # Add query image to the left
    query_col = 255*np.ones([grid_size_in_pixels, im_size])
    query_col[1:im_size+1, 0:im_size] = query_image

    # Concatenate
    grid = np.concatenate((query_col, 255*np.ones([grid_size_in_pixels, buff]), grid), 1)

    # Place grid image inside the Axes object
    ax.imshow(grid, cmap=plt.cm.gray)

    # Remove axes and borders
    ax.axis('off')
    ax.patch.set_visible('off')

    return ax


def shear(image, shear):
	"""Take a PyTorch Tensor object, shear it using 'shear' as the angle, then return
	the sheared image (also as a Tensor)"""

	xfm_to_PIL = transforms.ToPILImage()
	xfm_to_Tensor = transforms.ToTensor()
	sheared_image = xfm_to_Tensor( transforms.functional.affine( xfm_to_PIL(image), 
		angle=0, translate=(0,0), scale=1, shear=shear) )

	return sheared_image

def visualize_shear(ax, image, shear_list, ylabel=None, xlabel=False, title=None):
	"""Visualize the progression of shearing an image for several different shearing
	angles in a list

	Args:
		ax = A matplotlib Axes object
		image = The original image
		shear_list = The list of shearing angles
		ylabel = (Optional) The y-label for this axis will be the given argument
		xlabel = (Optional) The x-label for this axis will be the list of shears
		title = (Optional) Title of the axis

	Returns:
		ax = Finished plot
	"""

	# Create Numpy array for all the images
	c, h, w = image.shape
	n = len(shear_list)
	images = np.zeros([c, h, n*w+(n-1)*2])


	# Loop over the shear angles to show the progression of shears
	# Counter to hold x-coord that each image will start at
	x_start = 0
	for i, shear_angle in enumerate(shear_list):
		# Shear the image and concatenate it with the previous sheared images
		images[:,:,x_start:x_start+w] = 1.-shear(image, shear_angle)
	
		# Increment starting x-coord for next image
		x_start += w+2


	# Add image to the axis
	ax.imshow(images.squeeze(), cmap=plt.cm.binary)
	
	# Remove ticks and borders
	ax.set_yticks([])
	ax.set_xticks([])
	for spine in ax.spines:
		ax.spines[spine].set_visible(False)

	# Add labels/title if desired
	# Y labels will appear halfway up the image
	if ylabel is not None:
		ax.set_yticks([w/2])
		ax.set_yticklabels(ylabel)

	# X labels will appear halfway across each image
	if xlabel:
		ax.set_xticks(np.linspace(w/2, x_start-2-w/2, n))
		ax.set_xticklabels(['{}$^\circ$'.format(angle) for angle in shear_list])

	if title is not None:
		ax.set_title(title)

	return ax

def main():
	# Load MNIST training set
	train_set = datasets.MNIST('./datasets/mnist', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				]))

	im = train_set[0][0]

	shears = [-50, -40, -20, -30, -10, 0, 10, 20, 30, 40, 50]
	
	"""
	# Try plotting with plt.subplots()
	fig, axs = plt.subplots(10, 1)

	# Get a shear sample of each class
	for i in range(10):
		i_sample = next(sample[0] for sample in train_set if sample[1].item() == i)
		axs[i] = visualize_shear(axs[i], i_sample, shears)#, labels=i+1)

	fig.suptitle('MNIST images with varying shear angle')

	plt.show()
	"""

	# Try plotting with gridspec
	fig = plt.figure()
	gs = mpl.gridspec.GridSpec(10, 1, fig, wspace=0, hspace=0.1)

	# Get a shear sample for each class
	for i in range(10):
		title = None
		xlabel = False
		if i == 0:
			title = 'MNIST images with varying shear angle'
		elif i == 9:
			xlabel = True

		i_sample = next(sample[0] for sample in train_set if sample[1].item() == i)
		ax = plt.subplot(gs[i])
		ax = visualize_shear(ax, i_sample, shears, xlabel=xlabel, title=title, ylabel=str(i))

	# Add common axes labels
	fig.text(0.51, 0.03, 'Shear angle', ha='center')
	fig.text(0.13, 0.51, 'Class', va='center', rotation='vertical')
	plt.show()
if __name__ == '__main__':
	main()