import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math
import MyKMeans as MK

class MyRBFN(object):
    def __init__(self, hidden_shape = 50, sigma=1.0,k = 10):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None
        self.k = k

    def kernel_function(self, center, data_point):
        return np.exp(np.linalg.norm(center-data_point)**2/(-2*(self.sigma**2)))

    def calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self.kernel_function(center, data_point)
        G = np.concatenate([1 * np.ones((len(X), 1)),G], axis=1)
        return G

    def fit(self):
        X,Y = open_file()
        self.centers = MK.select_centers(X,self.k)
        G = self.calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        # X = np.concatenate([1 * np.ones((1, 1)),X], axis=1)
        G = self.calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

def open_file():
    path_to_file = "train4dAll_test.txt"
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
    p1,p2,p3 = 58, 1, 58
    model = MyRBFN(hidden_shape=p1, sigma=p2,k = p3)
    model.fit()
    y_pred = []
    for xi in x:
        y_pred.append(model.predict([xi]))
    plt.scatter(range(0,len(x)),y,c ='g')
    # plt.scatter(model.centers,c='R')
    plt.scatter(range(0,len(x)),y_pred , c ="red")
    plt.title("hidden_shape={}, sigma={},k = {}".format(str(p1),str(p2),str(p3)))
    plt.show()
