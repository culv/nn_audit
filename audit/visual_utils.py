"""
Author: Culver McWhirter
Date:   11 Sep 2018
Time:   13:48
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

import seaborn as sns

import sys
import os

def draw_vector2d(ax, v, c, other_kw=dict()):
    """Draw a 2d vector, v, on a given Matplotlib Axes object with color, c.

    Args:
        ax = Matplotlib Axes object
        v = The vector [x_tail, y_tail, dx, dy] (where dx and dy are the
            difference between head and tail along x and y directions)
        c = The color
        other_kw = Any other parameters for the vector (e.g. dict(headlength=0, linewidth=0.1) to
            draw without arrowheads and with line width 1)
    """

    ax.quiver(v[0], v[1], v[2], v[3], color=c, scale=1, angles='xy', scale_units='xy', **other_kw)


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

    cm = plt.cm.viridis

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

def show_neighbs_grid2(ax, query_image, neighbor_images, grid_size=(6,4), buff=10):
    """Show a grid of nearest neighbors in the training set. Query image is shown in the top left,
    and the neighbors are shown in a separate grid. Grid has rows and columns according to 'grid_size'

    Args:
        ax = A matplotlib Axes object to plot on
        query_image = The test image
        neighbor_images = The nearest training images
        grid_size = Tuple of the number of rows of images and the number of images per column, (rows, cols)
        buff = Number of pixels in between query image and neighbor images
    """

    # Cast to numpy arrays and floats
    query_image = np.array(query_image)
    neighbor_images = np.array(neighbor_images)

    # If images are 0-255 ints, convert to 0-1 floats
    if query_image.max()>1:
        query_image = np.array(query_image).astype(np.float)/255.
    if neighbor_images.max()>1:
        neighbor_images = np.array(neighbor_images).astype(np.float)/255.

    # Repeat to have 3 color channels
    query_image = np.repeat(query_image[:,:,None], 3, 2)
    neighbor_images = np.repeat(neighbor_images[:,:,:,None], 3, 3)

    # Grab number of nearest neighbors needed for the grid
    neighbor_images = neighbor_images[:grid_size[0]*grid_size[1]]

    # Image size in pixels
    im_size = query_image.shape[0]

    # Colors for query background and neighbor background
    coral = mpl.colors.to_rgb('xkcd:coral')
    blue = mpl.colors.to_rgb('xkcd:light blue')

    # Create blank grid, accounting for 1-pixel separation between images
    grid_size_pixels = (grid_size[0]*(im_size+1)+1, grid_size[1]*(im_size+1)+1)
    grid = np.tile(blue, (grid_size_pixels[0], grid_size_pixels[1], 1))

    # Loop over neighbor images and add them to the grid
    for row in range(grid_size[0]):
        # The row coordinate (starting at top, ending at bottom)
        r = row*(im_size+1)+1
        for col in range(grid_size[1]):
            # The column coordinate (starting left, ending right)
            c = col*(im_size+1)+1
            grid[r:r+im_size, c:c+im_size, :] = neighbor_images[row*grid_size[1]+col]
#            print(row, col, row*grid_size[1]+col)

    # Add query image to the left
    query_col = np.ones([grid_size_pixels[0], im_size+2, 3])

    # Repeat color channels for query image and add border
    query_image_ = np.tile(coral, (30, 30, 1))
    query_image_[1:-1, 1:-1, :] = query_image


    query_col[0:im_size+2, 0:, :] = query_image_

    # Concatenate
    grid = np.concatenate((query_col, np.ones([grid_size_pixels[0], buff, 3]), grid), 1)

    # Place grid image inside the Axes object
    ax.imshow(grid)

    # Remove axes and borders
    ax.axis('off')
    ax.patch.set_visible('off')

def shear(image, shear):
    """Take a PyTorch Tensor object, shear it using 'shear' as the angle, then return
    the sheared image (also as a Tensor)"""

    xfm_to_PIL = transforms.ToPILImage()
    xfm_to_Tensor = transforms.ToTensor()
    sheared_image = xfm_to_Tensor( transforms.functional.affine( xfm_to_PIL(image), 
        angle=0, translate=(0,0), scale=1, shear=shear) )

    return sheared_image

def visualize_shear(ax, images, shear_list, space=2, ylabel=None, xlabel=False, title=None):
    """Visualize the progression of shearing an image for several different shearing
    angles in a list

    Args:
        ax = A matplotlib Axes object
        images = A list of Pytorch tensors of images
        shear_list = The list of shearing angles
        ylabel = (Optional) The y-label for this axis will be the given argument
        xlabel = (Optional) The x-label for this axis will be the list of shears
        title = (Optional) Title of the axis
    """

    # Get dimensions of images
    n_i = len(images)
    c, h, w = images[0].shape

    # Number of shears
    n_s = len(shear_list)

    # Create blank grid, number rows=n_s, number columns=n_i
    grid_color = mpl.colors.to_rgb('xkcd:grey')
    grid = np.tile(grid_color, (n_i*(h+space)+space, n_s*(w+space)+space, 1))

    # Loop over images, shear, and add to grid
    for row in range(n_i):
        # The row coordinate (starting at top, ending at bottom)
        r = row*(h+space)+space
        for col in range(n_s):
            # The column coordinate (starting left, ending right)
            c = col*(w+space)+space

            # Shear the current image
            img = shear(images[row], shear_list[col])

            # Convert to Numpy, repeat color channels
            img = np.repeat(np.transpose(img.numpy(), (1,2,0)), 3, 2)

            # Add to grid
            grid[r:r+h, c:c+w, :] = img
            # print(row, col, row*n_s+col)

    # Add xticks, yticks, and labels at the centers of each image (height and width)
    ax.set_yticks(np.linspace(h/2+space, grid.shape[0]-(h/2+space), n_i))
    ax.set_xticks(np.linspace(w/2+space, grid.shape[1]-(w/2+space), n_s))

    ax.set_yticklabels(np.linspace(0,9,10).astype(np.int))
    ax.set_xticklabels(['{:d}$\degree$'.format(shear) for shear in shear_list])

    # Remove grid
    ax.grid(False)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.imshow(grid)


def main():
    # Load MNIST training set
    script_dir = os.path.dirname(os.path.realpath(__file__))
    mnist_dir = os.path.join(script_dir, '..', 'datasets', 'mnist')

    train_set = datasets.MNIST(mnist_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    im = train_set[0][0]
    shears = [-50, -40, -20, -30, -10, 0, 10, 20, 30, 40, 50]

    # Get a sample for each class
    samples = []
    for i in range(10):
        samples.append(next(sample[0] for sample in train_set if sample[1].item() == i))

    sns.set()
    fig, ax = plt.subplots()
    visualize_shear(ax, samples, shears, ylabel='Class', xlabel='Shear angle', title='MNIST with shears')

    plt.savefig(os.path.join(script_dir, '../figures/mnist_shears.png'), bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()