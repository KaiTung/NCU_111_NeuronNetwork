import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os

def raw_data_process():
    path_to_file = r'.\NN_HW1_DataSet\基本題\2Ccircle1.txt'
    data = []

    with open(path_to_file) as f:
        for i in f.readlines():
             i = i.split()
             data.append(i)
        
    return np.array(data).astype(float)

def Hard_limit(y):
    if y < 0:
        return 1
    else:
        return 2

def Training(max_epoch,lr):
    data = raw_data_process()
    
    
    n_samples = data.shape[0]
    n_features = data.shape[1] - 1
    
    x = np.concatenate([data[:,:-1], np.ones((n_samples, 1))], axis=1)    
    
    w = np.ones((1,n_features + 1))
    epoch = 0
    while epoch < max_epoch:
        detect = 0
        for i in range(n_samples):
            v  = np.dot(w,x[i])
            if data[i][2] != Hard_limit(v):
                detect += 1
                if v < 0:
                    w += lr * x[i].T
                else:
                    w -= lr * x[i].T
        
        if detect == 0:
            break

        epoch += 1
    
    return w
    
# def predict(x,w):
#     y=[]
#     for i in x:
#         y.append(np.dot(w,i))
    
#     return np.array(y)

def Plot(w,data):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlabel('X1')
    plt.ylabel('X2')
    #繪製資料
    idx_1 = [i for i in data[:,2]==1]
    ax.scatter(data[idx_1,0], data[idx_1,1], marker='o', color='g', label=1, s=20)
    idx_2 = [i for i in data[:,2]==2]
    ax.scatter(data[idx_2,0], data[idx_2,1], marker='x', color='r', label=2, s=20)
    #限制範圍
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    #取兩點畫線 w0x1 + w1x2 + w2 = 0; x1 = (-w1x2-w2)/w0 ; x2 = (-w0x1 - w2)/w1;(x1,x2)
    p1 = [0,(0*-w[0][0] -w[0][2])/w[0][1]]
    p2 = [(0*-w[0][1]-w[0][2])/w[0][0],0]
    plt.axline(p1,p2,label="w1",color='b')

    plt.legend(loc = 'upper right')
    plt.show()



# Hyper parameter
max_epoch = 50
lr = 0.8   

#Training
w = Training(max_epoch,lr)

data = raw_data_process()
#plot
Plot(w,data)