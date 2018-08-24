#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:59:37 2018

@author: culv
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_xy(x, y):
    f, ax = plt.subplots()
    ax.scatter(x, y, c='b')
    return ax

x = np.random.rand(1,5)
y = x

a = plot_xy(x, y)
a.scatter(0.5, 0.5, c='r')


a.annotate('test', np.array([0.5,0.5]), np.array([0.6,0.6]),
           arrowprops=dict(arrowstyle='->'))

plt.show()