import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets,uic
import math
import os
import sys

class My_UI(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_UI, self).__init__()
        self.ui = uic.loadUi('UI.ui', self) # 讀入.ui檔
        # 替combobox增加選項
        self.comboBox = self.findChild(QtWidgets.QComboBox,"comboBox")
        self.comboBox.addItems(["2Ccircle1","2Circle1","2Circle2","2CloseS","2CloseS2","2CloseS3","2cring","2CS","2Hcircle1","2ring","Perceptron1","Perceptron2"])
        # 取得lr
        self.lr = self.findChild(QtWidgets.QTextEdit,"textEdit")
        # 按鈕事件
        self.ui.Train_button.clicked.connect(self.Training)
        self.show() # Show the GUI
    
    def raw_data_process(self,path_to_file):
        data = []

        with open(path_to_file) as f:
            for i in f.readlines():
                i = i.split()
                data.append(i)
            
        return np.array(data).astype(float)

    def Hard_limit(self,x):
        if self.comboBox.currentText in ["Perceptron1","Perceptron2"]:
            if x < 0:
                return 0
            else:
                return 1
        else:
            if x < 0:
                return 2
            else:
                return 1

    def Training_w(self,max_epoch,lr):
        path_to_file = '.\\NN_HW1_DataSet\\基本題\\' + self.comboBox.currentText() + ".txt"
        data = self.raw_data_process(path_to_file)
        
        n_samples = data.shape[0]
        n_features = data.shape[1] - 1
        
        X = np.concatenate([data[:,:-1], np.ones((n_samples, 1))], axis=1)    
        Y = data[:,2]

        w = np.random.rand(n_features + 1)
        epoch = 0
        while epoch < max_epoch:
            detect = 0
            for i in range(n_samples):
                v  = np.dot(w,X[i])
                if Y[i] != self.Hard_limit(v):
                    detect += 1
                    if v < 0:
                        w += lr * X[i].T
                    else:
                        w -= lr * X[i].T
            

            if detect == 0:
                break

            epoch += 1
        
        return w

    def sigmoid(self,x):
        return 1/(1+math.exp(-x))

    # def Training_w_SGD(self,max_epoch,lr):
    #     path_to_file = '.\\NN_HW1_DataSet\\基本題\\' + self.comboBox.currentText() + ".txt"
    #     data = self.raw_data_process(path_to_file)
        
    #     n_samples = data.shape[0]
    #     n_features = data.shape[1] - 1
        
    #     X = np.concatenate([data[:,:-1], np.ones((n_samples, 1))], axis=1)    
    #     Y = data[:,2]
        
    #     w = np.random.rand(n_features + 1)

    #     epoch = 0

    #     while epoch < max_epoch:
    #         detect = 0
    #         E=0
    #         for i in n_samples:
    #             v = np.dot(w,X[i])
    #             E += 1/2 * (v - Y[i])**2
    #             delta_w = 

    #         w = w - lr * delta_w

    #         if detect == 0:
    #             break
            
    #         epoch += 1
    #     return w

    def Training(self):
        max_epoch = 200
        lr = float(self.lr.toPlainText())

        w= self.Training_w(max_epoch,lr)
        # w= self.Training_w_SGD(max_epoch,lr)

        self.Plot(w)
        
    # def predict(x,w):
    #     y=[]
    #     for i in x:
    #         y.append(np.dot(w,i))
        
    #     return np.array(y)

    def Plot(self,w):
        path_to_file = '.\\NN_HW1_DataSet\\基本題\\' + self.comboBox.currentText() + ".txt"

        class1 = 1
        class2 = 2

        if self.comboBox.currentText() in ["Perceptron1","Perceptron2"]:
            class1 = 1
            class2 = 0

        data = self.raw_data_process(path_to_file)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.xlabel('X1')
        plt.ylabel('X2')
        #繪製資料
        idx_1 = [i for i in data[:,2]==class1]
        ax.scatter(data[idx_1,0], data[idx_1,1], marker='s', color='b', label=class1, s=20)
        idx_2 = [i for i in data[:,2]==class2]
        ax.scatter(data[idx_2,0], data[idx_2,1], marker='x', color='g', label=class2, s=20)
        plt.legend(loc = 'best')
        #限制範圍
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        #取兩點畫線 w0x1 + w1x2 + w2 = 0; x1 = (-w1x2-w2)/w0 ; x2 = (-w0x1 - w2)/w1;(x1,x2)
        p1 = [0,(0*-w[0] -w[2])/w[1]]
        p2 = [(0*-w[1]-w[2])/w[0],0]
        plt.axline(p1,p2,label="w1",color='black')

        
        plt.show()

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = My_UI() # Create an instance of our class
    app.exec_() # Start the application


if __name__ == '__main__':
    main()