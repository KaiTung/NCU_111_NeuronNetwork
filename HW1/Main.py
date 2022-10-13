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

        #取得題號 QComboBox
        self.comboBox = self.findChild(QtWidgets.QComboBox,"comboBox")
        self.comboBox2 = self.findChild(QtWidgets.QComboBox,"comboBox_2")
        p1 = "./NN_HW1_DataSet/"
        p2 = os.listdir(p1) #加分題、基本題
        self.comboBox.addItems([p2[1]]) #只完成基本題
        self.comboBox_change()
        # self.comboBox2.addItems(["2Ccircle1","2Circle1","2CloseS","2CloseS2","2CloseS3","2cring","2CS","2Hcircle1","2ring","Perceptron1","Perceptron2"])
        # 取得lr QLineEdit
        self.lr = self.findChild(QtWidgets.QLineEdit,"lineEdit")
        # 取得maxEpoch QLineEdit
        self.max_epoch = self.findChild(QtWidgets.QLineEdit,"lineEdit_2")
        # 取得Acc QLineEdit
        self.Acc = self.findChild(QtWidgets.QLineEdit,"lineEdit_3")
        # 取得label_image
        self.label_image = self.findChild(QtWidgets.QLabel,"label_image")
        # 取得label_image2
        self.label_image2 = self.findChild(QtWidgets.QLabel,"label_image2")
        # 取得label_Acc
        self.label_Acc = self.findChild(QtWidgets.QLabel,"label_Acc")
        # 取得label_Valid_Acc
        self.label_Valid_Acc = self.findChild(QtWidgets.QLabel,"label_Valid_Acc")
        # 取得label_w
        self.label_w = self.findChild(QtWidgets.QLabel,"label_w")
        # 取得label_L1
        self.label_L1 = self.findChild(QtWidgets.QLabel,"label_L1")
        
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        
        # 先初始化label_image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.savefig("empty.png")

        self.label_image.setPixmap(QtGui.QPixmap("empty.png"))
        self.label_image.setScaledContents(True)
        self.label_image.show()

        self.label_image2.setPixmap(QtGui.QPixmap("empty.png"))
        self.label_image2.setScaledContents(True)
        self.label_image2.show() 

        
        # combobox改變觸發事件
        self.comboBox.currentIndexChanged.connect(self.comboBox_change)
        # Train按鈕事件
        self.ui.Train_button.clicked.connect(self.Training)
        self.show() # Show the GUI
    
    def comboBox_change(self):
        p0 = "./NN_HW1_DataSet/" + self.comboBox.currentText()
        self.comboBox2.clear()
        self.comboBox2.addItems(os.listdir(p0))

    def raw_data_process(self,path_to_file):
        data = []

        with open(path_to_file) as f:
            for i in f.readlines():
                i = i.split()
                data.append(i)
            
        return np.array(data).astype(float)

    def Hard_limit(self,x,Y):
        if x <= 0 : return Y[0]
        else: return Y[1]
            
    def sigmoid(self,v):
        return 1/(1 + math.exp(-v))
    
    def Training_single_perceptron(self,max_epoch,lr,data):
        
        n_samples = data.shape[0]
        n_features = data.shape[1] - 1
        
        train_samples = round(n_samples / 3 * 2);
        valid_samples = n_samples - train_samples;
        
        X = np.concatenate([-1 * np.ones((n_samples, 1)),data[:,:-1]], axis=1)    
        Y = data[:,2]
        
        rand_sampling = np.random.choice(range(n_samples),valid_samples,replace=False) #隨機取樣序列
        
        X_valid = X[rand_sampling]
        X_train = np.delete(X, rand_sampling, axis = 0)
        Y_valid = Y[rand_sampling]
        Y_train = np.delete(Y, rand_sampling, axis = 0)
        
        w = np.random.rand(n_features + 1)
        best_w = w
        best_Acc = 0
        epoch = 0
        while epoch < max_epoch:

            for i in range(train_samples):
                v  = np.dot(w,X_train[i].T)
                if Y_train[i] != self.Hard_limit(v,np.unique(Y)):
                    if v < 0:
                        w = w + lr * X_train[i]
                    else:
                        w = w - lr * X_train[i]
            
            #計算Acc
            wrong = 0
            for i in range(train_samples):
                v  = np.dot(w,X_train[i].T)
                if Y_train[i] != self.Hard_limit(v,np.unique(Y)):
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
        #計算Acc
        wrong = 0
        for i in range(train_samples):
            v  = np.dot(best_w,X_train[i].T)
            if Y_train[i] != self.Hard_limit(v,np.unique(Y)):
                wrong += 1

        best_Acc = (train_samples - wrong) / train_samples

        #計算Valid_Acc
        valid_wrong = 0
        for i in range(valid_samples):
            v  = np.dot(best_w,X_valid[i].T)
            if Y_valid[i] != self.Hard_limit(v,np.unique(Y)):
                valid_wrong += 1
        
        valid_Acc = (valid_samples - valid_wrong) / valid_samples
        
        self.label_Acc.setText("最佳訓練精準度: " + str(best_Acc))
        self.label_Valid_Acc.setText("最佳測試精準度: " + str(valid_Acc))
        self.label_w.setText("最佳解w: " + str(best_w))
        self.label_L1.setText("L1 : (" + str(round(best_w[0],8)) + ") + (" + str(round(best_w[1],8)) + ")x1 + (" + str(round(best_w[2],8)) + ")x2 = 0")
        return best_w,X_train,X_valid,Y_train,Y_valid


    def Training(self):
        path_to_file = '.\\NN_HW1_DataSet\\' + self.comboBox.currentText() + '\\' + self.comboBox2.currentText()
        data = self.raw_data_process(path_to_file)

        w,X_train,X_valid,Y_train,Y_valid = self.Training_single_perceptron(int(self.max_epoch.text()),float(self.lr.text()),data)
        self.Plot(w,data,X_train,X_valid,Y_train,Y_valid)


    def Plot(self,w,data,X_train,X_valid,Y_train,Y_valid):

        classes = np.unique(data[:,2])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel('x1')
        plt.ylabel('x2')

        #繪製訓練資料
        for c in range(len(classes)):
            idx_1 = [i for i in Y_train == classes[c]]
            ax.scatter(X_train[idx_1,1], X_train[idx_1,2], label = classes[c], s=20)

        #根據資料點自動縮放
        plt.autoscale(enable=True, axis='both', tight=None)
        #取兩點畫線 w0*-1 + w1x1 + w2x2 = 0; x1 = (w0 - w2x2)/w1 ; x2 = (w0 - w1x1)/w2;
        p1 = [0,(w[0] - w[1]*0)/w[2]]
        p2 = [(w[0] - w[2]*0)/w[1],0]
        plt.axline(p1,p2,label="L1",color='black')
        plt.legend(loc = 'best')
        plt.title("Training set")
        plt.savefig('pic1.png')

        fig.clf()
        ax = fig.add_subplot(111)
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        #繪製測試資料
        for c in range(len(classes)):
            idx_1 = [i for i in Y_valid == classes[c]]
            ax.scatter(X_valid[idx_1,1], X_valid[idx_1,2], label = classes[c], s=20)
    
        #根據資料點自動縮放
        plt.autoscale(enable=True, axis='both', tight=None)
        #取兩點畫線 w0*-1 + w1x1 + w2x2 = 0; x1 = (w0 - w2x2)/w1 ; x2 = (w0 - w1x1)/w2;
        p1 = [0,(w[0] - w[1]*0)/w[2]]
        p2 = [(w[0] - w[2]*0)/w[1],0]
        plt.axline(p1,p2,label="L1",color='black')
        plt.legend(loc = 'best')
        plt.title("Validation Set")
        plt.savefig('pic2.png')

        self.label_image.setPixmap(QtGui.QPixmap("pic1.png"))
        self.label_image.show()

        self.label_image2.setPixmap(QtGui.QPixmap("pic2.png"))
        self.label_image2.show()

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = My_UI() # Create an instance of our class
    app.exec_() # Start the application

    #刪掉圖片
    if os.path.exists("pic1.png"):
        os.remove("pic1.png")
    if os.path.exists("pic2.png"):
        os.remove("pic2.png")


if __name__ == '__main__':
    main()