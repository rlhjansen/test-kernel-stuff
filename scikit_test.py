import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn import datasets, svm

np.random.seed(0)

X, y = make_circles(n_samples=10, factor=.3, noise=.05)

X = np.array([[10,-40], [10,100], [9,5], [10,12], [9,50], [9,13], [10,0]])
y = np.array([1,1,0,1, 0, 0,1])
print(X)
print(y)


def k_n_m2(xn, xm, thetas=[1,0.17,0,0]):
    k = thetas[0] * np.exp(-(thetas[1]/2.) * (np.sqrt((xn-xm).T @ (xn-xm)))) # + thetas[2] + thetas[3] * (xn.T @ xm)
    return k

def computeK2(X):
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i, j] = k_n_m2(X[i,:], X[j,:])
    return K

def computeK2_test(X_train, X_test):
    K = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            K[i, j] = k_n_m2(X_train[j,:], X_test[i,:])
    return K



n_sample = len(X)

y = y.astype(np.float)

X_train = X[:y.shape[0]-1]
y_train = y[:y.shape[0]-1]
X_test = X[y.shape[0]-1:]
y_test = y[y.shape[0]-1:]

print(X_test)
# fit the model
for x_val2 in range(-10,-5):
    # for kernel in ('linear', 'rbf', 'poly'):
    X[5,1] = x_val2
    for kernel in ['rbf', 'precomputed']:
        # clf = svm.SVC(kernel=kernel)
        # print(vars(clf))
        # {'kernel': 'rbf', 'gamma': 'scale', 'tol': 0.001, 'C': 1.0, 'nu': 0.0, 'epsilon': 0.0, 'shrinking': False, 'probability': False, 'cache_size': 500, 'class_weight': None, 'verbose': False, 'max_iter': -1, 'random_state': None}
        # exit(0)
        clf = svm.SVC(gamma=1, kernel=kernel, probability=True, C=1.0)
        print(vars(clf))
        if kernel == 'precomputed':
            gram_matrix = computeK2(X_train)

            print(gram_matrix.shape)
            clf.fit(gram_matrix, y_train)
        else:
            clf.fit(X_train, y_train)

        plt.figure()
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)

        # Circle out the test data
        plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                    zorder=10, edgecolor='k')

        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        plt.xlim(y_min, y_max)
        plt.ylim(y_min, y_max)

        # XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        XX, YY = np.mgrid[y_min:y_max:50j, y_min:y_max:50j]
        if kernel == "precomputed":
            GRAMMIE_BOI = computeK2_test(X_train, np.c_[XX.ravel(), YY.ravel()])
            Z = clf.predict(GRAMMIE_BOI)
            print("GRAMMIE_BOI!")
        else:
            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

        plt.title(kernel)
    plt.show()
    break
