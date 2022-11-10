import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math

class MyRBFN(object):
    def __init__(self, hidden_shape, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def kernel_function(self, center, data_point):
        return np.exp(np.linalg.norm(center-data_point)**2/(-2*self.sigma**2))

    def calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self.kernel_function(
                        center, data_point)
        return G

    # def select_centers(self, X):
    #     random_args = np.random.choice(len(X), self.hidden_shape)
    #     centers = X[random_args]
    #     return centers

    def euclidean_distance(self,x1,x2):
        distance = 0
        for dim in range(len(x1)):
            distance += pow(x1[dim] - x2[dim],2)
        distance = pow(distance,0.5)
        return distance

    def select_centers(self,x):
        k = 3
        S = []
        # frist kth x as centers
        for i in range(k):
            S.append([x[i]])

        z = []
        for i in range(k):
            z.append(x[i])

        done = 1
        last_z = []
        while done:
            last_z = z[:]
            #clustering
            for x_i in x:
                #caculate dis from x to z_i
                dis = []
                for z_i in z:
                    dis.append(self.euclidean_distance(z_i,x_i))
                #find min dis
                S[dis.index(min(dis))].append(x_i)
            #update z
            for i in range(len(z)):
                z[i] = sum(S[i]) / len(S[i])
            print("z={}".format(z))

            if last_z == z:
                done = 0
                print("DONE")
        return z


    def fit(self, X, Y):
        self.centers = self.select_centers(X)
        G = self.calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self.calculate_interpolation_matrix(X)
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
    # print(y_pred)
    # plotting 1D interpolation
    # plt.plot(y, 'b-', label='real')
    # plt.plot(y_pred, 'r-', label='fit')
    # plt.legend(loc='upper right')
    # plt.title('Interpolation using a RBFN')
    # plt.show()
        
