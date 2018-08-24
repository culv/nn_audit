"""
Author: Culver McWhirter
Date:   13 Mar 2018
Time:   21:30
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import axes3d
from sklearn import decomposition

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class NN_Auditor:
    def __init__(self, embed_dim, neuron_data_fn):
        self.neuron_data_fn = neuron_data_fn # filename containing neuron data
        self.neuron_data = np.array([]) # neuron data array
        self.embed_data = np.array([]) # embedded data array
        self.data_labels = np.array([]) # data labels
        self.embed_min = 0. # min value of all embedded vectors
        self.embed_max = 1. # max value
        self.embed_basis = np.array([]) # transformation matrix to go from neuron data -> embedded vectors 
        self.embed_dim = embed_dim # lower dimension to embed into
    
    # normalize values to be in [0,1]
    def normalize(self, data):
        self.embed_min = np.min(data)
        self.embed_max = np.max(data)
        self.embed_data = (data - self.embed_min) / (self.embed_max - self.embed_min)
        return self.embed_data
    
    def load_neuron_data(self):
        self.neuron_data = np.loadtxt(self.neuron_data_fn)
        return self.neuron_data
        
    def truncated_SVD(self, data):
        svd = decomposition.TruncatedSVD(n_components = self.embed_dim)
        # zero out mean
        data = data - np.mean(data, 0)
        self.embed_data = svd.fit_transform(data)
        self.embed_basis = svd.components_.T
        return self.embed_data, self.embed_basis

# scale and visualize the embedding vectors (supports 2d and 3d plots)
# INPUTS:    dat = the data
#            lab = the labels            
#            title = desired title of the plot
#            plot_type = either plot as 'scatter' or 'digit'
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
    return ax

def main():
    nn_auditor = NN_Auditor(2, 'mnist_100neurons_data.gz')
    neurons = nn_auditor.load_neuron_data()
    
    pca_neurons, pca_basis = nn_auditor.truncated_SVD(neurons)
    pca_neurons = nn_auditor.normalize(pca_neurons)

    mnist = input_data.read_data_sets('MNIST/', one_hot=True)
    labels = np.argmax(mnist.validation.labels, 1)

    ax = plot_embedding(pca_neurons, labels, plot_type='digit')    
    
    plt.show()

if __name__ == '__main__':
    # main()
    nn_auditor = NN_Auditor(2, 'mnist_100neurons_data.gz')
    neurons = nn_auditor.load_neuron_data()
#    neurons = neurons[:,0:100]
    
    pca_neurons, pca_basis = nn_auditor.truncated_SVD(neurons)
    pca_neurons = nn_auditor.normalize(pca_neurons)

    mnist = input_data.read_data_sets('MNIST/', one_hot=True)
    labels = np.argmax(mnist.validation.labels, 1)

    ax = plot_embedding(pca_neurons, labels, plot_type='digit')    
    
    plt.show()