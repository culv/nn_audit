import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

# generate data matrix
X = np.random.rand(10,5)

# embedding dimension
k = 2

# data matrix normalized to mean 0
X_norm = X - np.mean(X,0)

# first, use eigen decomposition on cov(X) to do PCA
cov = np.matmul(X_norm.T, X_norm)
eigval, eigvec = np.linalg.eig(cov)

# sort eigenvalues in descending order
top_k_eigs = np.argsort(-eigval)[0:k]

eig_basis = eigvec.T[top_k_eigs].T


# second, use SVD on X to do PCA
U, sigma, VT = np.linalg.svd(X_norm)
top_k_sing_val = np.argsort(-sigma)[0:k]
svd_basis = VT[top_k_sing_val].T

              
eig_pca = np.matmul(X, eig_basis)
svd_pca = np.matmul(X, svd_basis)

svd = decomposition.TruncatedSVD(n_components=2)
skl_svd_pca = svd.fit_transform(X_norm)

pca = decomposition.PCA(n_components=2)
skl_eig_pca = pca.fit_transform(X)


plt.figure(1)
plt.subplot(221)
plt.scatter(eig_pca[:,0], eig_pca[:,1])
plt.title('Homemade Eig PCA')

plt.subplot(222)
plt.scatter(svd_pca[:,0], svd_pca[:,1], c='r')
plt.title('Homemade SVD PCA')

plt.subplot(223)
plt.scatter(skl_eig_pca[:,0], skl_eig_pca[:,1], c='m')
plt.title('SciKit PCA')

plt.subplot(224)
plt.scatter(skl_svd_pca[:,0], skl_svd_pca[:,1], c='g')
plt.title('SciKit TruncSVD')
plt.show()