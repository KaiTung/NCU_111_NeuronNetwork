import numpy as np
import matplotlib.pyplot as plt
from numpy import random

class MyRBFN(object):
    def __init__(self, hidden_shape, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(np.linalg.norm(center-data_point)**2/(-2*self.sigma**2))

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

if __name__ == "__main__":

    path_to_file = "train4dAll.txt"
    data = []
    with open(path_to_file) as f:
        for i in f.readlines():
            i = i.split()
            data.append(i)
    data = np.array(data).astype(float)
    n_samples = data.shape[0]
    n_features = data.shape[1]
    # x = np.concatenate([1 * np.ones((n_samples, 1)),data[:,:-1]], axis=1)
    x = data[:,:-1]
    y = data[:,n_features-1]

    # fitting RBF-Network with data
    model = MyRBFN(hidden_shape=10, sigma=1.)
    model.fit(x, y)
    y_pred = model.predict(x)
    print(y_pred)
    # plotting 1D interpolation
    # plt.plot(y, 'b-', label='real')
    # plt.plot(y_pred, 'r-', label='fit')
    # plt.legend(loc='upper right')
    # plt.title('Interpolation using a RBFN')
    # plt.show()
        
