import numpy as np

import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

import sys
import os

# Temporarily add parent directory to path
sys.path.append('./..')
from audit.visual_utils import draw_vector2d

def main():

	X = np.array([[2, 2, 4, 4, 5, 5, 6, 8, 8, 8, 9, 9, 8, 9],
		          [4, 8, 2, 9, 1, 6, 3, 1, 3, 5, 2, 7, 8, 9]]).T

	sns.set()

	m1 = np.median(X[:,0])

	m2 = np.median(X[0:7, 1])

	m3 = np.median(X[8:, 1])

	fig, ax = plt.subplots()

	# Darker background for searched area
	rect = mpl.patches.Rectangle((0, 0), 7, 4, color='c', alpha=0.5, zorder=0)
	ax.add_artist(rect)

	# Circle around neighbors
	circ = mpl.patches.Circle((5, 2), 1.2, color='yellow', alpha=0.8, zorder=1)
	ax.add_artist(circ)

	# Draw lines for medians
	draw_vector2d(ax, [m1, -2, 0, 20], 'r', dict(zorder=1))
	draw_vector2d(ax, [m1, m2, -20, 0], 'g', dict(zorder=1))
	draw_vector2d(ax, [m1, m3, 20, 0], 'b', dict(zorder=1))

	# Original data	
	ax.scatter(X[:,0], X[:,1], s=35, c='c', edgecolors='k', alpha=1, zorder=2)

	# Query point
	ax.scatter(5, 2, s=35, c='r', edgecolors='k', alpha=1, zorder=2)


	ax.set_aspect('equal')
	ax.set_xlim([0, 10])
	ax.set_ylim(ax.get_xlim())
	ax.set_xlabel('$x_1$')
	ax.set_ylabel('$x_2$', rotation=0)

	plt.savefig('kd_tree2.png', bbox_inches='tight')


	plt.show()


if __name__ == '__main__':
	main()