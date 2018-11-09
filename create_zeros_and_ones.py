import h5py
import numpy as np

def main():
	# Load training set
	f_train = h5py.File('./datasets/neurons/shear_mnist_test_100neurons.hdf5', 'r')

	# Create a new file for just zeros and ones
	f_zo = h5py.File('./datasets/neurons/shear_zeros_ones_test.hdf5', 'w')

	# Get the indices of training samples with a class label of 0 or 1
	idcs = np.argwhere(np.array(f_train['labels']) < 2).squeeze()

	# Loop over the datasets in f_train and grab the data corresponding to 0s and 1s, and
	# copy over to f_zo
	for dset in f_train.keys():
		f_zo[dset] = np.array(f_train[dset])[idcs]

if __name__ == '__main__':
	main()