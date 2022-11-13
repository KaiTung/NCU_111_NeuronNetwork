import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math
from MyKMeans import *

class MyRBFN(object):
    def __init__(self, hidden_shape = 40, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def kernel_function(self, center, data_point):
        return np.exp(np.linalg.norm(center-data_point)**2/(-2*(self.sigma**2)))

    def calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self.kernel_function(center, data_point)
        G = np.concatenate([1 * np.ones((len(X), 1)),G], axis=1)
        return G

    def euclidean_distance(self,x1,x2):
        distance = 0
        for dim in range(len(x1)):
            distance += pow(x1[dim] - x2[dim],2)
        distance = pow(distance,0.5)
        return distance

    def cmp_list(self,l1,l2):
        for i in range(len(l1)):
            for j in range(len(l1[i])):
                if l1[i][j] != l2[i][j]:return False
        return True

    def fit(self,k):
        X,Y = open_file()
        n_samples = X.shape[0]
        # X = np.concatenate([1 * np.ones((n_samples, 1)),X], axis=1) # add bias
        self.centers = select_centers(X,k)
        G = self.calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        # X = np.concatenate([1 * np.ones((1, 1)),X], axis=1)
        G = self.calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

def open_file():
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
    return x,y

if __name__ == "__main__":
    x,y = open_file()
    # fitting RBF-Network with data
    model = MyRBFN(hidden_shape=20, sigma=3)
    for i in range(20,21):
        model.fit(k=i)
        y_pred = []
        for xi in x:
            y_pred.append(model.predict([xi]))
            print(xi,model.predict([xi]))
        # plotting 1D interpolation
        # plt.plot(y, 'b-', label='real')
        # plt.plot(y_pred, 'r-', label='fit')
        # plt.legend(loc='upper right')
        # plt.title('RBFN k = {}'.format(i))
        # plt.show()
        
