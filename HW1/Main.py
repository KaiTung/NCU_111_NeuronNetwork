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
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.data_type = {"01":["Perceptron1","Perceptron2"],
                          "12":["2ring","2Hcircle1","2CS","2cring","2CloseS3","2CloseS2","2CloseS","2Cricle1","2Ccircle1","2Circle2"],
                          "124":["2Circle2"]}
        #取得題號 QComboBox
        self.comboBox = self.findChild(QtWidgets.QComboBox,"comboBox")
        # 替combobox增加選項
        self.comboBox.addItems(["2Ccircle1","2Circle1","2Circle2","2CloseS","2CloseS2","2CloseS3","2cring","2CS","2Hcircle1","2ring","Perceptron1","Perceptron2"])
        # 取得lr QLineEdit
        self.lr = self.findChild(QtWidgets.QLineEdit,"lineEdit")
        # 取得maxEpoch QLineEdit
        self.max_epoch = self.findChild(QtWidgets.QLineEdit,"lineEdit_2")
        # 取得Acc QLineEdit
        self.Acc = self.findChild(QtWidgets.QLineEdit,"lineEdit_3")
        
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
        if str(self.comboBox.currentText()) in self.data_type["01"]:
            if x <= 0 : return 0
            else: return 1
        elif str(self.comboBox.currentText()) in self.data_type["12"]:
            if x <= 0 : return 1
            else: return 2
            
    def sigmoid(self,v):
        return 1/(1 + math.exp(-v))
    
    def PLA(self,max_epoch,lr):
        path_to_file = '.\\NN_HW1_DataSet\\基本題\\' + self.comboBox.currentText() + ".txt"
        data = self.raw_data_process(path_to_file)
        
        n_samples = data.shape[0]
        n_features = data.shape[1] - 1
        
        X = np.concatenate([-1 * np.ones((n_samples, 1)),data[:,:-1]], axis=1)    
        Y = data[:,2]

        
        w = np.random.rand(n_features + 1)
        best_w = w
        epoch = 0
        while epoch < max_epoch:

            for i in range(n_samples):
                v  = np.dot(w,X[i].T)
                if Y[i] != self.Hard_limit(v):
                    if v < 0:
                        w = w + lr * X[i]
                    else:
                        w = w - lr * X[i]
            
            
            #計算Acc
            wrong = 0
            for i in range(n_samples):
                v  = np.dot(w,X[i].T)
                if Y[i] != self.Hard_limit(v):
                    wrong += 1
                
            #全對停止
            if wrong == 0:
                best_w = w
                print("完美切割!")
                break
            
            # 達標
            Acc = (n_samples - wrong) / n_samples
            if Acc >= float(self.Acc.text()):
                print("Acc=",Acc)
                print("達到目標精準度")
                best_w = w
                
            epoch += 1
            
        return best_w


    def Training(self):
        
        w = self.PLA(int(self.max_epoch.text()),float(self.lr.text()))

        self.Plot(w)


    def Plot(self,w):
        path_to_file = '.\\NN_HW1_DataSet\\基本題\\' + self.comboBox.currentText() + ".txt"

        class1 = 1
        class2 = 2

        if str(self.comboBox.currentText()) in self.data_type["01"]:
            class1 = 0
            class2 = 1
        elif str(self.comboBox.currentText()) in self.data_type["124"]:
            class1 = 1
            class2 = 2
            class3 = 4

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
        if str(self.comboBox.currentText()) in self.data_type["124"]:
            idx_3 = [i for i in data[:,2]==class3]
            ax.scatter(data[idx_3,0], data[idx_3,1], marker='^', color='r', label=class3, s=20)
        plt.legend(loc = 'best')
        #根據資料點自動縮放
        plt.autoscale(enable=True, axis='both', tight=None)
        #取兩點畫線 w0*-1 + w1x1 + w2x2 = 0; x1 = (w0 - w2x2)/w1 ; x2 = (w0 - w1x1)/w2;
        p1 = [0,(w[0] - w[1]*0)/w[2]]
        p2 = [(w[0] - w[2]*0)/w[1],0]
        plt.axline(p1,p2,label="w1",color='black')

        
        plt.show()

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = My_UI() # Create an instance of our class
    app.exec_() # Start the application


if __name__ == '__main__':
    main()