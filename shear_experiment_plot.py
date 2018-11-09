import matplotlib as mpl
mpl.use('TkAgg') # Use TkAgg backend to prevent segmentation fault
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

def main():
	# Seaborn aesthetic
	sns.set()

	# bar chart params
	width = 0.1
	x = np.array([0.3,0.7])


	# full experiment
	err = np.array([0.111626*100, 0.123455*100])
	var = np.array([263.687213, 1759.205608])
	lab = ['20-PCA', '9-LDA']

	fig, err_ax = plt.subplots()


	err_ax.tick_params(axis='y', labelcolor='b')
	err_ax.set_xlim([0, 1])
	err_ax.set_ylim([0, 100])
	err_ax.set_xticks(x)
	err_ax.set_yticks(np.linspace(0, 100, 11))
	err_ax.set_xticklabels(lab)
	err_ax.set_ylabel('Class error (%)')

	var_ax = err_ax.twinx()

	# Plot variance
	var_ax.bar(x+width/2, var, width, color='r')


	# Plot error, but plot it on the variance axis to avoid grid lines on top
	# (will need to rescale)
	var_ax.bar(x-width/2, 2000/100*err, width, color='b')

	var_ax.tick_params(axis='y', labelcolor='r')
	var_ax.set_yticks(np.linspace(0, 2000, 11))
	var_ax.set_ylabel('Shear variance ($degrees^2$)')
	var_ax.set_title('Average class error and shear variance of 500 nearest neighbors from train set relative to test set (ALL CLASSES)')



	# zeros and ones experiment
	err2 = np.array([0.00137589*100, 0.00175267*100])
	var2 = np.array([186.01322, 1993.4087])
	lab2 = ['20-PCA', '1-LDA']

	fig2, err_ax2 = plt.subplots()


	err_ax2.tick_params(axis='y', labelcolor='b')
	err_ax2.set_xlim([0, 1])
	err_ax2.set_ylim([0, 1])
	err_ax2.set_xticks(x)
	err_ax2.set_yticks(np.linspace(0, 1, 11))
	err_ax2.set_xticklabels(lab2)
	err_ax2.set_ylabel('Class error (%)')

	var_ax2 = err_ax2.twinx()

	# Plot variance
	var_ax2.bar(x+width/2, var2, width, color='r')


	# Plot error, but plot it on the variance axis to avoid grid lines on top
	# (will need to rescale)
	var_ax2.bar(x-width/2, 2200*err2, width, color='b')

	var_ax2.tick_params(axis='y', labelcolor='r')
	var_ax2.set_yticks(np.linspace(0, 2200, 11))
	var_ax2.set_ylabel('Shear variance ($degrees^2$)')
	var_ax2.set_title('Average class error and shear variance of 500 nearest neighbors from train set relative to test set (ZEROS AND ONES)')


	plt.show()

if __name__ == '__main__':
	main()