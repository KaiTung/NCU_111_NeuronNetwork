import numpy as np
import matplotlib.pyplot as plt
import MyKMeans as MK

class MyRBFN(object):
    def __init__(self, hidden_shape = 1475, sigma=2,k = 58):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None
        self.k = k

        self.x = None
        self.y = None

    #基底函數
    def kernel_function(self, center, data_point):
        return np.exp(np.linalg.norm(center-data_point)**2/(-2*(self.sigma**2)))

    #虛擬反置矩陣
    def calculate_interpolation_matrix(self, X):
        phi_of_x = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                phi_of_x[data_point_arg, center_arg] = self.kernel_function(center, data_point)
        phi_of_x = np.concatenate([1 * np.ones((len(X), 1)),phi_of_x], axis=1)
        return phi_of_x

    def fit(self):
        self.centers = MK.select_centers(self.x,self.k)
        phi_of_x = self.calculate_interpolation_matrix(self.x)
        self.weights = np.dot(np.linalg.pinv(phi_of_x), self.y)

    def predict(self, X):
        phi_of_x = self.calculate_interpolation_matrix(X)
        predictions = np.dot(phi_of_x, self.weights)
        return predictions

    def read_training_data(self,path_to_file):
        data = []
        with open(path_to_file) as f:
            for i in f.readlines():
                i = i.split()
                data.append(i)
        data = np.array(data).astype(float)
        n_features = data.shape[1]
        self.x = data[:,:-1]
        self.y = data[:,n_features-1]
        

if __name__ == "__main__":
    # fitting RBF-Network with data
    p1,p2,p3 = 1475, 2, 58
    model = MyRBFN(hidden_shape=p1, sigma=p2,k = p3)
    model.read_training_data("train4dAll.txt")
    model.fit()
    y_pred = []
    for xi in model.x:
        y_pred.append(model.predict([xi]))
    plt.scatter(range(0,len(model.x)),model.y,c ='g')
    # plt.scatter(model.centers,c='R')
    plt.scatter(range(0,len(model.x)),y_pred , c ="red")
    plt.title("hidden_shape={}, sigma={},k = {}".format(str(p1),str(p2),str(p3)))
    plt.show()
