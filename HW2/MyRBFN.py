import numpy as np
import matplotlib.pyplot as plt

class MyRBFN(object):

    def __init__(self,h_layers,sigma):
        self.h_layers = h_layers
        self.sigma = sigma
        self.weights = None
        self.centers = None
        
    def basis_function(self,center,data):
        return np.exp((- np.linalg.norm(data - center)**2)/2*self.sigma**2)

    def K_means(self,x,k):
        n=np.size(x,axis=0)
        dim=np.size(x,axis=1)
        Center = np.random.random([k,dim])
        maxiter=50
        count_iter=0
        while count_iter<maxiter:
            count_iter+=1
            dist=np.array(np.zeros([n,k]))
            for ic in range(k):
                c=Center[ic,:]
                dist[:,ic] = np.linalg.norm(x-c,axis=1)
            old_Center=np.copy(Center)
            Center_index=dist.argmin(axis=1)
            for ic in range(k):
                mu=np.mean(x[(Center_index==ic),:],axis=0)
                Center[ic,:]=mu
             
            th=np.linalg.norm(Center-old_Center)          
            if th<np.spacing(1):
                break
                
        for ic in range(k):
            c=Center[ic,:]
            dist[:,ic] = np.linalg.norm(x-c,axis=1)
        Center_index=dist.argmin(axis=1)
            
        return Center,Center_index,dist

    def calculate(self,x):
        PHI = np.zeros((len(x), self.h_layers))
        for data_point_arg, data_point in enumerate(x):
            for center_arg, center in enumerate(self.centers):
                PHI[data_point_arg, center_arg] = self.basis_function(center, data_point)
        return PHI

    def get_PHI_of_x(self,x):

        return

    def predict(self, x):
        PHI_of_x = self.get_PHI_of_x(x)
        F_of_x = np.dot(PHI_of_x, self.weights)
        return F_of_x

    def training(self,x,y,epoch):
        #選出中心點
        self.centers,_,_ = self.K_means(x,k = 1)

        #計算
        ans = self.calculate(x)
        #更新weights
        self.weights = np.dot(np.linalg.pinv(ans), y)
    
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
    x = np.concatenate([1 * np.ones((n_samples, 1)),data[:,:-1]], axis=1)
    y = data[:,n_features-1]

    model = MyRBFN(h_layers=20,sigma=1)
    model.training(x,y,epoch = 5)
    pre  = model.predict(x)

    # plotting 1D interpolation
    plt.plot(y, 'b-', label='real')
    plt.plot(pre, 'r-', label='fit')
    plt.legend(loc='upper right')
    plt.title('Interpolation using a RBFN')
    plt.show()

