#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:41:08 2018

@author: culv
"""

from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

kNN = neighbors.NearestNeighbors(n_neighbors=25)

X = np.random.rand(100,2)
kNN.fit(X)
test_pt = np.array([[0.5,0.5]])
dist, neighbs = kNN.kneighbors(test_pt)

f, ax = plt.subplots()

X_neighbs = X[neighbs[0]]

X_else = np.delete(X, neighbs[0], axis=0)

ax.scatter(X[:,0],X[:,1],c='b')
ax.scatter(test_pt[:,0], test_pt[:,1],c='r')
ax.scatter(X_neighbs[:,0], X_neighbs[:,1], c='orange')