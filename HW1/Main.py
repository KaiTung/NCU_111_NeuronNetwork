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
                          "12":["2ring","2Hcircle1","2CS","2cring","2CloseS3","2CloseS2","2CloseS","2Cricle1","2Ccircle1"]
                          }
        #取得題號 QComboBox
        self.comboBox = self.findChild(QtWidgets.QComboBox,"comboBox")
        self.comboBox2 = self.findChild(QtWidgets.QComboBox,"comboBox_2")
        self.comboBox.addItems(["基本題","加分題"])
        self.comboBox2.addItems(["2Ccircle1","2Circle1","2CloseS","2CloseS2","2CloseS3","2cring","2CS","2Hcircle1","2ring","Perceptron1","Perceptron2"])
        # 取得lr QLineEdit
        self.lr = self.findChild(QtWidgets.QLineEdit,"lineEdit")
        # 取得maxEpoch QLineEdit
        self.max_epoch = self.findChild(QtWidgets.QLineEdit,"lineEdit_2")
        # 取得Acc QLineEdit
        self.Acc = self.findChild(QtWidgets.QLineEdit,"lineEdit_3")
        # 取得label_image
        self.image_label = self.findChild(QtWidgets.QLabel,"label_image")
        # 取得label_Acc
        self.label_Acc = self.findChild(QtWidgets.QLabel,"label_Acc")
        # 取得label_Valid_Acc
        self.label_Valid_Acc = self.findChild(QtWidgets.QLabel,"label_Valid_Acc")
        # 取得label_w
        self.label_w = self.findChild(QtWidgets.QLabel,"label_w")
        
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        
        # 先初始化image_label
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.savefig("empty.png")
        self.image_label.setPixmap(QtGui.QPixmap("empty.png"))
        self.image_label.setScaledContents(True)
        self.image_label.show() 
        
        # combobox改變觸發事件
        self.comboBox.currentIndexChanged.connect(self.comboBox_change)
        # Train按鈕事件
        self.ui.Train_button.clicked.connect(self.Training)
        self.show() # Show the GUI
    
    def comboBox_change(self):
        if self.comboBox.currentText() == "基本題":
            self.comboBox2.clear()
            self.comboBox2.addItems(["2Ccircle1","2Circle1","2CloseS","2CloseS2","2CloseS3","2cring","2CS","2Hcircle1","2ring","Perceptron1","Perceptron2"])
        elif self.comboBox.currentText() == "加分題":
            self.comboBox2.clear()
            self.comboBox2.addItems(["2Circle2","4satellite-6","5CloseS1","8OX","C3D","C10D","IRIS","Number","perceptron3","perceptron4","wine","xor"])

    def raw_data_process(self,path_to_file):
        data = []

        with open(path_to_file) as f:
            for i in f.readlines():
                i = i.split()
                data.append(i)
            
        return np.array(data).astype(float)

    def Hard_limit(self,x):
        if str(self.comboBox2.currentText()) in self.data_type["01"]:
            if x <= 0 : return 0
            else: return 1
        elif str(self.comboBox2.currentText()) in self.data_type["12"]:
            if x <= 0 : return 1
            else: return 2
            
    def sigmoid(self,v):
        return 1/(1 + math.exp(-v))
    
    def Training_single_perceptron(self,max_epoch,lr):
        path_to_file = '.\\NN_HW1_DataSet\\' + self.comboBox.currentText() + '\\' + self.comboBox2.currentText() + ".txt"
        data = self.raw_data_process(path_to_file)
        
        n_samples = data.shape[0]
        n_features = data.shape[1] - 1
        
        train_samples = n_samples // 3 * 2;
        valid_samples = n_samples - train_samples;
        
        X = np.concatenate([-1 * np.ones((n_samples, 1)),data[:,:-1]], axis=1)    
        Y = data[:,2]
        
        rand_sampling = np.random.choice(range(n_samples),valid_samples,replace=False) #隨機取樣序列
        
        X_valid = X[rand_sampling]
        # X_train = X
        X_train = np.delete(X, rand_sampling, axis = 0)
        Y_valid = Y[rand_sampling]
        # Y_train = Y
        Y_train = np.delete(Y, rand_sampling, axis = 0)
        
        w = np.random.rand(n_features + 1)
        best_w = w
        best_Acc = 0
        epoch = 0
        while epoch < max_epoch:

            for i in range(train_samples):
                v  = np.dot(w,X_train[i].T)
                if Y_train[i] != self.Hard_limit(v):
                    if v < 0:
                        w = w + lr * X_train[i]
                    else:
                        w = w - lr * X_train[i]
            
            #計算Acc
            wrong = 0
            for i in range(train_samples):
                v  = np.dot(w,X_train[i].T)
                if Y_train[i] != self.Hard_limit(v):
                    wrong += 1
                
            Acc = (train_samples - wrong) / train_samples
        
            if Acc == 1: #全對停止
                best_Acc = Acc
                best_w = w
                print("完美切割!")
                break
            elif  Acc > best_Acc: # 精準度提升
                best_Acc = Acc
                best_w = w

            if Acc >= float(self.Acc.text()): #達標停止
                break
                
            epoch += 1
            
        #計算Valid_Acc
        valid_wrong = 0
        for i in range(valid_samples):
            v  = np.dot(w,X_valid[i].T)
            if Y_valid[i] != self.Hard_limit(v):
                valid_wrong += 1
        
        valid_Acc = (valid_samples - valid_wrong) / valid_samples
        
        self.label_Acc.setText("最佳訓練精準度: " + str(best_Acc))
        self.label_Valid_Acc.setText("最佳測試精準度: " + str(valid_Acc))
        self.label_w.setText("最佳解w: " + str(best_w))
        return best_w


    def Training(self):
        
        w = self.Training_single_perceptron(int(self.max_epoch.text()),float(self.lr.text()))
        
        self.Plot(w)


    def Plot(self,w):
        path_to_file = '.\\NN_HW1_DataSet\\' + self.comboBox.currentText() + '\\' + self.comboBox2.currentText() + ".txt"

        class1 = 1
        class2 = 2

        if str(self.comboBox2.currentText()) in self.data_type["01"]:
            class1 = 0
            class2 = 1

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
        #根據資料點自動縮放
        plt.autoscale(enable=True, axis='both', tight=None)
        #取兩點畫線 w0*-1 + w1x1 + w2x2 = 0; x1 = (w0 - w2x2)/w1 ; x2 = (w0 - w1x1)/w2;
        p1 = [0,(w[0] - w[1]*0)/w[2]]
        p2 = [(w[0] - w[2]*0)/w[1],0]
        plt.axline(p1,p2,label="w1",color='black')

        plt.savefig('pic.png')
        
        self.image_label.setPixmap(QtGui.QPixmap("pic.png"))
        self.image_label.show() 

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = My_UI() # Create an instance of our class
    app.exec_() # Start the application


if __name__ == '__main__':
    main()