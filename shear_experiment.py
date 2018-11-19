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

from matplotlib import offsetbox

from scipy.misc import imresize

import seaborn as sns

import h5py

from networks.mnist_convnet import RandomChoiceShear, ConvNet
from audit.dim_reduction import PCA, LDA
from audit.visual_utils import shear, show_neighbs_grid2, plot_embedding
from audit.nn_audit_utils import kNN

from tqdm import tqdm

import pickle

import os
import sys

# Load MNIST train and test sets as global variables
script_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = os.path.join(script_dir, 'datasets', 'mnist')
mnist_train = datasets.MNIST(mnist_dir, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        ]))

mnist_test = datasets.MNIST(mnist_dir, train=False, download=True,
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
    ax.grid(True)
    ax.set_axisbelow(True)

def evaluate_dim_red(dataset, knn_model, dim_red_model, p=20):
    """Evaluate the dimensionality reduction method for neural network auditing purposes. For each test
    example, calculate the:
    
        1) percentage of nearest training neighbors that do not have the same prediction as the 
            test example (roughly a measure of how well model preserves class separation)
        2) standard deviation (in degrees) of nearest train set neighbor shears relative to
            test example shear (roughly a measure of how well model preserves style encoding)

    and average these two measures over the entire test set.

    Args:
        dataset = The HDF5 file of neuron activations, predictions, etc. for the sheared test set
        knn_model = A kNN model fit to the training set neuron activations
        dim_reduce = A valid dimensionality reduction object with a 'project' method (eg. PCA, LDA, etc.)
            that is fit to the training set

    Returns:
        avg_predict_err = The average percentage of nearest neighbor training labels different from test
            labels
        avg_shear_std = The average standard deviation of nearest neighbor training shears from test shears
    """

    # Initialize shear_var
    avg_predict_err = 0
    avg_shear_std = 0

    # Length of test set
    n = dataset['shears'].shape[0]

    # Batch size for knn searches
    bs = 256

    # Number of batches
    n_batches = np.ceil(n/bs).astype(np.uint)

    # Loop over batches of shears and neuron activations
    b_start = 0
    for b in tqdm(range(n_batches)):
        # Get end index of the batch
        b_end = min(n, b_start+bs)

        # Get batch of activations (and project them) and shears and labels
        # Make sure to convert shears to floats since they are stored as 8-bit integers
        a = dim_red_model.project(dataset['activations'][b_start:b_end,:], p=p)
        s = np.array(dataset['shears'][b_start:b_end]).astype(np.float)
        pr = np.array(dataset['predictions'][b_start:b_end])

        # Get nearest neighbors for neuron batch
        neighbs, _ = knn_model.query(a)

        # Get the shears corresponding to neighbors
        s_neighbs = np.array(dataset['shears']).astype(np.float)[neighbs[:,1:]]
        pr_neighbs = np.array(dataset['labels'])[neighbs[:,1:]]

        # Calculate batch class error and batch variance
        b_err = np.mean(pr_neighbs != pr[:,np.newaxis])
        b_std = np.sqrt(np.mean((s_neighbs-s[:,np.newaxis])**2))

        # Weight batch calculations and then add to totals
        b_weight = (b_end-b_start)/n
        avg_predict_err += b_weight*b_err
        avg_shear_std += b_weight*b_std

        # Increment batch start to begin from where previous batch ended
        b_start = b_end

    return avg_predict_err, avg_shear_std


def main():
    """
    # Get query activations
    #   5: q_id=779, q_shear=25.3
    #   3: q_id=334, q_shear=-38.4
    #   7 with bar: q_id=9009, q_shear=8.6
    #   1 generic, q_id=984
    q_id = 779
    q_shear = 25.3

    q = get_query(q_id, q_shear)

    # Get query image
    q_image = shear(mnist_test.test_data[q_id][None,:,:], q_shear).squeeze().numpy()

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
    neighbor_images = np.concatenate(neighbor_images, 0)

    sns.set()

    # Plot the query image, nearest training images, and histogram of shears

    fig, ax = plt.subplots()

    show_neighbs_grid2(ax, q_image, neighbor_images, grid_size=(4,3), buff=10)

    # Get the properties of Axes title, and use for text
    font_props = ax.title.get_font_properties()#dict(color='k', weight='bold')
    ax.text(0.105, 0.968, 'Query ({:.1f}$\degree$ shear)'.format(q_shear), fontproperties=font_props, transform=plt.gcf().transFigure)
    ax.text(0.545, 0.968, 'Neighbors', fontproperties=font_props, transform=plt.gcf().transFigure)
    
    plt.tight_layout()
#    plt.savefig(os.path.join(script_dir, 'figures/quantitative_7_neighbors.png'), bbox_inches='tight')
    # plt.show()
    # sys.exit()

    fig2, ax2 = plt.subplots()
    
    shear_histogram(ax2, q_shear, neighbor_shears, title='Relative frequency of shears (500 nearest neighbors)')

    # Put image within histogram
    if False:
        back_color = mpl.colors.to_rgb('xkcd:coral')
        q_image_ = np.tile(back_color, (30,30,1))
        q_image_[1:-1,1:-1,:] = np.repeat(q_image[:,:,None], 3, 2)
        q_image_ = imresize(q_image_, 300)

        img_loc = (10, 0.5)
        imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(q_image_, cmap=plt.cm.gray),
                img_loc, pad=0)
        ax2.add_artist(imagebox)
       
        font_props = ax2.title.get_font_properties()
        ax2.text(img_loc[0]-17, img_loc[1]+0.2, 'Query ({:.1f}$\degree$ shear)'.format(q_shear), fontproperties=font_props)
    
    ax2.set_ylim(0, 1)
    plt.savefig(os.path.join(script_dir, 'figures/quantitative_5_hist.png'), bbox_inches='tight')

    plt.show()

    """
    # Evaluate PCA predict error and shear standard deviation

    # Do PCA
    """
    pca = PCA('./models/DELETE_THIS_conv_shear_test_PCA.hdf5')
    if False:
        pca.run(test_neurons['activations'])
        pca.save()
    else:
        pca.load()

    # Do LDA

    lda = LDA('./models/DELETE_THIS_conv_shear_test_LDA.hdf5')
    if False:
        lda.run(test_neurons['activations'], test_neurons['predictions'])
        lda.save()
    else:
        lda.load()

    stats = dict(d=[], avg_predict_err=[], avg_shear_std=[])

    for j in range(1, 46):

        # Fit kNN to the PCA'ed data
        knn = kNN(None, k=501)
        knn.fit(pca.project(test_neurons['activations'], j))

        avg_predict_err, avg_shear_std = evaluate_dim_red(test_neurons, knn, pca, p=j)
        print(j)
        print('\tavg predict err:\t{:.2f}'.format(100*avg_predict_err))
        print('\tavg shear std:\t{:.2f} degrees'.format(avg_shear_std))
        stats['d'].append(j)
        stats['avg_predict_err'].append(avg_predict_err)
        stats['avg_shear_std'].append(avg_shear_std)

    with open('pca_stats.pickle', 'wb') as f:
        pickle.dump(stats, f)
    """

    with open('pca_stats.pickle', 'rb') as f:
        pca_stats = pickle.load(f)

    with open('lda_stats.pickle', 'rb') as f:
        lda_stats = pickle.load(f)


    # Make figure for nearest neighbor prediction error and shear standard deviation
    lda_kw = dict(color='r', marker='v', linewidth=1, markersize=5.5)
    pca_kw = dict(color='b', marker='o', linewidth=1, markersize=4)

    sns.set()

    fig3, ax3 = plt.subplots()#figsize=(4,4))

    ax3.plot(pca_stats['d'], np.array(pca_stats['avg_predict_err']), **pca_kw)
    ax3.plot(lda_stats['d'], np.array(lda_stats['avg_predict_err']), **lda_kw)
    ax3.set_xlim(0, 46)
    ax3.set_xticks(np.arange(0, 46, 5))
    ax3.set_yticks(np.arange(0, .71, .05), minor=True)
    ax3.grid(which='minor')
    ax3.set_ylabel('Prediction match')
    ax3.set_xlabel('Projection dimension')
    ax3.set_title('Average prediction match (500 neighbors)')
    ax3.legend(['PCA', 'LDA'])

    plt.savefig(os.path.join(script_dir, 'figures/avg_predict_error.png'), bbox_inches='tight')

    fig4, ax4 = plt.subplots()#figsize=(4,4))

    ax4.plot(pca_stats['d'], pca_stats['avg_shear_std'], **pca_kw)
    ax4.plot(lda_stats['d'], lda_stats['avg_shear_std'], **lda_kw)
    ax4.set_xlim(0, 46)
    ax4.set_ylim(0, 45)
    ax4.set_xticks(np.arange(0, 46, 5))
    ax4.set_yticks(np.arange(0, 46, 5), minor=True)
    ax4.grid(which='minor')
    ax4.set_ylabel('Shear deviation (degrees)')
    ax4.set_xlabel('Projection dimension')
    ax4.set_title('Average shear deviation (500 neighbors)')
    ax4.legend(['PCA', 'LDA'])

    plt.savefig(os.path.join(script_dir, 'figures/avg_shear_std.png'), bbox_inches='tight')

    plt.show()



if __name__ == '__main__':
	main()