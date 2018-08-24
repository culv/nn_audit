import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import axes3d
from sklearn import (manifold, datasets)

# scale and visualize the embedding vectors
# INPUTS:    X = the embedding
#            y = the labels            
#            title = desired title of the plot
#            plot_type = either plot embedded vectors as 'scatter' or 'digit'
def plot_embedding(X, y, title=None, plot_type='scatter', samples=True):
    # normalize all embedding vectors
    x_min, x_max = np.min(X,0), np.max(X,0)
    X = (X - x_min) / (x_max - x_min)

    # get dimension of embedding vectors
    dim = X.shape[1]

    if dim == 3:
        fig = plt.figure(dpi=60)
        ax = axes3d.Axes3D(fig) #fig.gca(projection='3d')
        #ax.set_axis_off()

        # loop over all vectors in embedding (same as # of digits)
        if plot_type == 'digit':
            for i in range(X.shape[0]):
                # plot digit as a string, at the location determined by the embedding X
                ax.text(X[i,0], X[i,1], X[i,2], str(y[i]),
                    color=plt.cm.tab10(y[i] / 10.), # color determined from Set1 color map
                    fontdict={'weight': 'bold', 'size': 9}) # format font
    
        elif plot_type == 'scatter':
            ax.scatter(X.T[0,:], X.T[1,:], X.T[2,:], c = y)

    if dim == 2:
        # create 2d figure
        f, ax = plt.subplots()
        ax.scatter(0,0,s=0)
        # loop over all vectors in embedding (same as # of digits)
        if plot_type == 'digit':
            for i in range(X.shape[0]):
                # plot digit as a string, at the location determined by the embedding X
                # NOTE: depending on version of matplotlib, tab10 colormap may be Vega10
                try:
                    ax.text(X[i,0], X[i,1], str(y[i]), # position and string corresponding to digit
                    color=plt.cm.tab10(y[i] / 10.), # color determined from Set1 color map
                    fontdict={'weight': 'bold', 'size': 9}) # format font
                except:
                    ax.text(X[i,0], X[i,1], str(y[i]),
                            color=plt.cm.Vega10(y[i] / 10.), # color determined from Set1 color map
                            fontdict={'weight': 'bold', 'size': 9}) # format font
        elif plot_type == 'scatter':
            ax.scatter(X.T[0,:], X.T[1,:], c=y)
        if samples:
            # show digit images on plot
            if hasattr(offsetbox,  'AnnotationBbox'): # will only work with matplotlib versions past v1.0
                # initialize shown images locations array with upper right corner of plot
                shown_images = np.array([[1.,1.]])
                # loop over all digits
                for i in range(X.shape[0]):
                    # calculate squared distance between current image's embedding vector and all others that have been displayed
                    dist = np.sum((X[i] - shown_images) ** 2, 1)
                    # if the smallest squared distance is below threshold, don't print it (to ensure plot isn't overcrowded)
                    if np.min(dist) < 4e-3:
                        continue
                    # otherwise, add embedding vector to array of shown images
                    shown_images = np.r_[shown_images, [X[i]]]
                    # and display image of digit at the embedding vector location
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(digits.images[i],  cmap=plt.cm.gray_r),
                        X[i])
                    ax.add_artist(imagebox)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return ax


def pca_eig(data, dim, zero_mean=True):
    if zero_mean:
        mean = np.mean(data, 0) # calculate mean of rows of data
        norm_data = data - mean # normalize data to zero mean
    cov = np.matmul(norm_data.T, norm_data)
    eigval, eigvec = np.linalg.eig(cov)
    top_eigs = np.argsort(-eigval)[0:dim] # sort -eigval to get descending order!
    pca_basis = eigvec.T[top_eigs].T
    pca_embedding = np.matmul(norm_data, pca_basis)
    return pca_basis, pca_embedding

digits = datasets.load_digits(n_class = 10)
X = digits.data
y = digits.target

X_test, X_train = np.split(X, [10])
y_test, y_train = np.split(y, [10])


emb_basis, pca_embed = pca_eig(X_train, 2)

# calculate embedding vectors for new data
test_embed = np.matmul(X_test, emb_basis)
# normalize test embedding vectors
x_min, x_max = np.min(pca_embed,0), np.max(pca_embed,0)
test_embed = (test_embed - x_min) / (x_max - x_min)

# plot training embedding
a = plot_embedding(pca_embed, y_train, 'PCA Embedding for dim=2', plot_type='digit', samples=False)
# plot new data with annotation
p = 6
a.scatter(test_embed[p,0], test_embed[p,1], s=30, c='r')
a.annotate('new data projection', test_embed[p], test_embed[p]+0.25,
           arrowprops=dict(arrowstyle='->'))
plt.show()