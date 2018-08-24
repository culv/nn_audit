#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:41:32 2018

@author: culv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from textwrap import wrap
from mpl_toolkits.mplot3d import axes3d
from sklearn import decomposition, neighbors
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

    
# normalize values to be in [0,1]
# INPUTS: data = data to normalize
# OUTPUTS: normalized data
def normalize(data):
    # normalize data
    d_min = np.min(data)
    d_max = np.max(data)
    return (data - d_min) / (d_max - d_min)
 
# load np.array from .gz file
# INPUTS:    fname = filename where data is saved
# OUTPUTS:   numpy array of data
def load_np_array(fname):
    return np.loadtxt(fname)

# run truncated SVD on data (similar to PCA but more robust to numerical
# precision and more efficient)
# INPUTS:   data = data to undergo dimensionality reduction
#           embed_dim = dimension to reduce data to
#           zero_out_mean = (True/False) subtract mean of features from each training example
# OUTPUTS:  embed_data = transformed data
#           embed_basis = top embed_dim eigenvectors used to transform data
def truncated_SVD(data, embed_dim, zero_out_mean=True, get_var=True):
    svd = decomposition.TruncatedSVD(n_components = embed_dim)
    if zero_out_mean: # zero out mean
        data = data - np.mean(data, 0)
    embed_data = svd.fit_transform(data)
    embed_basis = svd.components_.T
    if get_var: # get variance information of PCA components
        var_percent = svd.explained_variance_ratio_
        return embed_data, embed_basis, var_percent
    else:
        return embed_data, embed_basis

# save the embedded data and embedding basis
# INPUTS:   basis = numpy array of eigenvectors to transform data to reduced dimension
#           save_fn = filename to save to
def save_embedding_map(basis, save_fn):
    np.savetxt(save_fn, basis)

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

# fit k-nearest neighbors model to neuron data
# INPUTS:   fit_data = data to fit model to
#           n_neighbs = number of neighbors
# OUTPUTS:  kNN = sklearn NearestNeighbors obect
def kNearestNeighbs(fit_data, n_neighbs):
    kNN = neighbors.NearestNeighbors(n_neighbors = n_neighbs)
    kNN.fit(fit_data)
    return kNN

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

# visualize % of kNN that fall in the same class for each test image        
def neighbor_acc_visual(neighbor_acc_array, baseline, dims_vec, images):
    num_subplots = neighbor_acc_array.shape[0]
    cols = 2
    rows = np.ceil(num_subplots/cols).astype('int64')
    if num_subplots == 1:
        fig, ax = plt.subplots()
        ax = [ax]
    else:
        fig, ax = plt.subplots(rows, cols)
        ax = ax.flatten()
    fig.suptitle('Percentage of matching neighbors (k=500) and Variance Preserved by PCA for each dimension p') 
    for i in range(num_subplots):
        neighb_line = ax[i].scatter(dims, 100*neighbor_acc_array[i, :], c='g', s=7)
        base_line = ax[i].scatter(dims, 100*baseline, c='b', s=5)
        ax[i].set_xlabel('dimension p')
        if (i==2) or (i==3):
            ax[i].set_ylabel('% Variance Preserved/% Matching Neighbors')
        ax[i].set_xlim(0,100)
        ax[i].set_ylim(0,100)
        ax[i].grid()
        ax[i].set_xticks(np.arange(0,101,1), minor=True)
        ax[i].set_yticks(np.arange(0,101,10))
        ax[i].grid(which='minor')
        
        imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(1-images[i].reshape(28,28), cmap=plt.cm.gray_r),
                (5,95), pad=0)
        ax[i].add_artist(imagebox)
        
    fig.legend((neighb_line, base_line), ('% kNN with same label', '% variance preserved'))
    return fig, ax
   
# show grids of nearest neighbor images for chosen dimensions
def show_neighbs_grid(nn_tensor, dims_list):
    grid_size_in_ims = np.sqrt(nn_tensor.shape[1]).astype('int64')
    num_neighbs = nn_tensor.shape[1]-1
    num_subplots = nn_tensor.shape[0]
    imsize = np.sqrt(nn_tensor.shape[2]).astype('int64')
    cols = 3
    rows = np.ceil(num_subplots/2).astype('int64')
    # make subplots
    if num_subplots == 1:
        fig, ax = plt.subplots()
        ax = [ax]
    else:
        fig, ax = plt.subplots(rows, cols)
        ax = ax.flatten()

    #loop over chosen dimensions    
    for i in range(num_subplots):
        # format grid of images with test image in upper left corner    
        true_grid_size = int(grid_size_in_ims*imsize + (grid_size_in_ims+2)*2)
        grid = np.ones([true_grid_size, true_grid_size])
        for j in range(grid_size_in_ims): # y-coord of grid
            jx = j*(imsize+2)+2
            for k in range(grid_size_in_ims): # x-coord of grid
                kx = k*(imsize+2)+2
                grid[kx:kx+imsize, jx:jx+imsize] = 1-nn_tensor[i, (j*grid_size_in_ims)+k].reshape(imsize, imsize)
        # add border around test image and whole grid
        grid[imsize:imsize+4, 0:imsize+4] = 0.5
        grid[0:imsize+4, imsize:imsize+4] = 0.5
        grid = np.pad(grid, ((4,4),(4,4)), 'constant', constant_values=((0.5,0.5),(0.5,0.5)))        
        ax[i].imshow(grid, cmap=plt.cm.gray_r)
        ax[i].set_title('p={}'.format(dims_list[i]))
    for this_ax in ax:
        this_ax.axis('off') # remove axes
        this_ax.patch.set_visible('off') # remove borders
    fig.suptitle('{} Nearest Neighbors of Test Image in p-dimensional Neuron Space'.format(num_neighbs))
    return fig, ax


if __name__ == '__main__':

    # TRAINING STAGE:
    # - load or generate neuron data for training set
    # - find embedding basis for specified dimensions
    # - save embedding basis
    training = False
    if training:
        print('[ ] Start training...')
        # load train and test neuron data
        print('\t[ ] Loading training set neuron data...')
        train_neurons = load_np_array('mnist_100neurons_train_set_neuron_data.gz')
        # chop off classifications layer
        train_neurons = train_neurons[:,0:100]
        print('\t[*] DONE LOADING NEURON DATA')

        # max embedding dimension (contains the basis for dim p={1,2,..., n-1} in
        # its first p columns)
        max_dim = train_neurons.shape[1]-1

        print('\t[ ] Finding TruncatedSVD basis...')
        embedding, basis, var_ratio = truncated_SVD(train_neurons, max_dim)
        cum_var_ratio = np.cumsum(var_ratio)
        save_fn = '{}dim_basis.gz'.format(max_dim)
        var_fn = 'PCA_model_variance.gz'
        save_embedding_map(basis, save_fn)
        save_embedding_map(cum_var_ratio, var_fn)
        print('\t[*] SAVED BASIS TO {}'.format(save_fn))
        print('[*] DONE TRAINING')


#        basis = np.loadtxt('3dim_basis.gz')
#        embedding = np.matmul(train_neurons, basis)

#        # normalize data for plotting
#        embedding = normalize(embedding)
#    
#        # load labels and images from MNIST
#        mnist = input_data.read_data_sets('MNIST/', one_hot=True)
#        # convert labels from one-hot to digit value
#        train_labels = np.argmax(mnist.train.labels, 1)
#    
#        # plot embedding visualization
#        print('[ ] Plotting embedding...')
#        fig, ax = plot_embedding(embedding, train_labels, '3-d embedding of training set\'s 100-d neuron activations', plot_type='digit')
#        ax.set_xlabel('principal component 1')
#        ax.set_ylabel('principal component 2')
#        ax.set_zlabel('principal component 3')
#        plt.show()
#        print('[*] Finished plot')
    

    # TESTING STAGE:
    # - load labels and images from MNIST in TensorFlow
    # - load or generate neuron data for training and test set
    # - load embedding basis for specified dimensions
    # - transform training and test data into each embedding dimension
    # - find k-nearest neighbors in each embedding dimension
    # - create k-nearest neighbors visualizations
    
    testing = True
    if testing:
        print('[ ] Start testing...')
        print('\t[ ] Loading MNIST data...')
        # load labels and images from MNIST
        mnist = input_data.read_data_sets('MNIST/', one_hot=True)
        # convert labels from one-hot to digit value
        train_labels = np.argmax(mnist.train.labels, 1)
        train_images = mnist.train.images
 
        test_labels = np.argmax(mnist.test.labels, 1)
        test_images = mnist.test.images

        print('\t[*] DONE LOADING')

        
        # load training and test set neuron data
        print('\t[ ] Loading neuron data...')
        train_neurons = load_np_array('mnist_100neurons_train_set_neuron_data.gz')
        test_neurons = load_np_array('mnist_100neurons_test_set_neuron_data.gz')
        # chop off classifications layer
        train_neurons = train_neurons[:,0:100]
        test_neurons = test_neurons[:,0:100]
        print('\t[*] DONE LOADING')

        cum_var_ratio = load_np_array('PCA_model_variance.gz')

        # load embedding basis
        print('\t[ ] Loading embedding basis...')
        basis_file = '{}dim_basis.gz'.format(99)
        basis = load_np_array(basis_file)
        print('\t[*] DONE LOADING')
        
        # transform neuron data into basis, and find k-nearest neighbors
        neighbs_list = []
        k = 500
        test_idx = [12, 734, 33, 508, 809, 914]
        print('\t[ ] Finding k-nearest neighbors...')
        dims = np.arange(1,100,1)
        for i, dim in enumerate(dims):
            train_neurons_embed = np.matmul(train_neurons, basis[:, 0:dim])
            test_neurons_embed = np.matmul(test_neurons, basis[:, 0:dim])

            # if 1-dim embedding, put train set embedding into 2D np.array()            
            if dim == 1:
                train_neurons_embed = train_neurons_embed.reshape(1,-1).T
            # if only 1 test point, grab its embedding and reshape into 2D np.array()
            if len(test_idx) == 1:
                test_point_embed = test_neurons_embed[test_idx].reshape(1,-1)

            else:
                test_points_embed = test_neurons_embed[test_idx]

            print('\t\t[ ] Fitting kNN model for {}-dim...'.format(dim))            
            knn = kNearestNeighbs(train_neurons_embed, n_neighbs=k)

            dist, neighbs = knn.kneighbors(test_points_embed)
            neighbs_list.append(neighbs)
            print('\t\t[*] FOUND NEIGHBORS FOR {}-dim'.format(dim))
        print('\t[*] DONE FINDING NEIGHBORS')
        
        print('\t[ ] Grabbing nearest neighbor images and calculating accuracy...')
        # the test images
        test_points_images = test_images[test_idx]
        # list to hold neighbor images and accuracies
        nn_images_list = []
        neighbor_acc = []
        # shifter for manipulating training set neighbor indexing
        shifter = train_labels.shape[0]*np.matmul(
                np.array([np.arange(0,len(test_idx),1)]).T, 
                np.ones([1, k]))
        # get labels of test points
        test_point_labels = test_labels[test_idx]
        # create comparer matrix to compare test and neighbor labels
        comparer = np.matmul(
                np.array([test_point_labels]).T,
                np.ones([1,k]))

        # sample dimensions
        sample_dims = [1,2,3,5,10,20]
        for i, neighb_idx_array in enumerate(neighbs_list):

            if (i+1) in sample_dims:
                # get neighbor images
                neighb_images = train_images[neighb_idx_array[:,0:35].flatten()].reshape([6,35,784])
                # add test image to front
                neighb_images_with_test = np.concatenate((test_points_images.reshape([6,1,784]), neighb_images), axis=1)
                nn_images_list.append(neighb_images_with_test)
            
            # start making one hot vector of neighb_idx_array
            one_hot_idxs = (neighb_idx_array + shifter).flatten().astype('int64')
            one_hot = np.zeros(len(test_idx)*train_labels.shape[0])
            one_hot[one_hot_idxs] = 1
            one_hot = one_hot.reshape(len(test_idx), train_labels.shape[0])
            # reshape one hot vector into 2d array and multiply by test_labels
            neighb_labels = one_hot*(train_labels+1) # add one so that you can remove all zeros and get just 500 neighbor's labels, then subtract it again
            neighb_labels = neighb_labels[neighb_labels!=0].reshape(len(test_idx),k)-1

            # compute accuracy for all test points
            acc = np.array([np.mean((comparer==neighb_labels), 1)]).T
            if i == 0:
                neighbor_acc = acc
            else:
                neighbor_acc = np.concatenate((neighbor_acc, acc), axis=1)

        print('\t[*] DONE')

        print('\t[ ] Plotting neighbor accuraccy/PCA variance...')
        # plot accuracy visual
        fig, ax = neighbor_acc_visual(neighbor_acc, cum_var_ratio, dims, test_points_images)
        print('\t[*] DONE')
        # reorganize neighbor images so that same test image but different
        # dimensions are displayed on the same plot
        nn_images_list = np.array(nn_images_list)
        nn_images_list = np.transpose(nn_images_list, (1,0,2,3))
        print('\t[ ] Plotting neighbor grids...')
        # plot neighbor images visual
        placeholder = []
        for i in range(nn_images_list.shape[0]):
            placeholder.append(show_neighbs_grid(nn_images_list[i], sample_dims))
        print('\t[*] DONE')
        
        
#        print('\t[ ] Creating nearest neighbors visualization...')        
#        fig, ax = visualize_neighbors(test_image, neighbor_acc, neighbor_images_list, dims)
#        # plot using tight_layout() to avoid overlapping titles/figures
#        plt.tight_layout(h_pad=5)
#        plt.show()
#        print('\t[*] DONE')

        print('[*] DONE TESTING')