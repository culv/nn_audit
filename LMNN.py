#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:22:26 2018

@author: culv
"""

import numpy as np
import matplotlib.pyplot as plt
from shogun import LMNN, RealFeatures, MulticlassLabels

def make_cov_ellipse(cov):
    import matplotlib.patches as patches
    import scipy.linalg as linalg
    
    # ellipse is centered at (0,0)
    mu = np.array([0,0])
    
    # get eigvals/eigvecs of covariance matrix
    w,v = linalg.eigh(cov)
    
    # normalize eigvecs
    u = v[0]/linalg.norm(v[0])
    
    # angle in degrees
    angle = 180.0/np.pi*np.arctan(u[1]/u[0])

    # Gaussian ellipse at 2 standard deviation
    ellipse = patches.Ellipse(mu, 2*w[0]**0.5, 2*w[1]**0.5, 180+angle, color='orange', alpha=0.3)
    return ellipse

def plot_2d(fig, ax, dat, labs, xlab, ylab, title, alpha):
    X0, X1, X2 = dat[labs==0], dat[labs==1], dat[labs==2]
    
    ax.scatter(X0[:,0], X0[:,1], c='g', s=50, alpha=alpha)
    ax.scatter(X1[:,0], X1[:,1], c='r', s=50, alpha=alpha)
    ax.scatter(X2[:,0], X2[:,1], c='b', s=50, alpha=alpha)
    
    lim = 1.2*np.max(np.abs(dat))
    
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_aspect('equal')
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)


x = np.array([[0,0],[-1,0.1],[0.3,-0.05],[0.7,0.3],[-0.2,-0.6],[-0.15,-0.63],[-0.25,0.55],[-0.28,0.67]])
y = np.array([0,0,0,0,1,1,2,2])

# create new figure/axes
fig, ax = plt.subplots(2,2)

# flatten Axis object array into 1d
ax = ax.flatten()

# plot original data
plot_2d(fig, ax[0], x, y, 'x1', 'x2', 'original', 1)
# plot covariance
ellipse = make_cov_ellipse(np.eye(2))
ax[0].add_artist(ellipse)

# wrap features and labels in Shogun objects
features = RealFeatures(x.T)
labels = MulticlassLabels(y.astype(np.float64))

# number of target neighbors and iters
k = 1
iters = 1000

# create LMNN object
lmnn = LMNN(features, labels, k)

# set initial transform (identity matrix/Euclidean distance)
init_xfm = np.eye(2)

# run multipass LMNN
for i in range(3):
    # set number of iterations and train
    lmnn.set_maxiter(iters)
    lmnn.train(init_xfm)

    # get linear transform
    L = lmnn.get_linear_transform()

    # save prev transformed data for plotting
    x_ref = np.matmul(x, init_xfm)

    # update initial transform for next pass
    init_xfm = np.matmul(L, init_xfm)

    # plot transformed data
    plot_2d(fig, ax[i+1], x_ref, y, 'x1', 'x2', 'after {} LMNN'.format((i+1)*iters), 0.3) # previous, faded
    plot_2d(fig, ax[i+1], np.matmul(L, x_ref.T).T, y, 'x1', 'x2', 'after {} iterations LMNN'.format((i+1)*iters), 1) # transformed
    
    # plot covariance
    ellipse = make_cov_ellipse(np.eye(2))
    ax[i+1].add_artist(ellipse)

plt.show()