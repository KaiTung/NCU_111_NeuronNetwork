import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyQt5
from PyQt5 import QtWidgets, QtGui, uic
from MyRBFN import MyRBFN
import math
import sys
import os

class car(object):
    def __init__(self,x = 0,y = 0,theta = 0,phi = 90,b = 3):
        self.x = x
        self.y = y
        self.theta = theta  # -40 < theta < 40
        self.phi = phi   # -90 < phi < 270
        self.b = b      #car length
        
    def car_move(self,theta):
        # 更新 x,y,phi
        # 要先把Degree轉radian
        radian_theta = math.radians(theta)
        radian_phi = math.radians(self.phi)
        self.x = self.x + math.cos(radian_phi + radian_theta) + math.sin(radian_theta) * math.sin(radian_phi)
        self.y = self.y + math.sin(radian_phi + radian_theta) - math.sin(radian_theta) * math.cos(radian_phi)
        self.phi = self.phi - (math.asin(2 * math.sin(radian_theta)/self.b) * (180.0 / math.pi)) # radian to degree
    
        
class MyGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
         
        #初始化地圖
        self.map=[]
        #讀取&儲存座標點
        with open('軌道座標點.txt','r') as f:
            for i in f.read().splitlines():
                self.map.append(np.array(i.split(',')).astype(int))
        #畫布宣告一次就好
        self.fig = plt.figure()
        self.plot_new_graph(0,0,90)
        # 取得label_image
        self.label_image = self.findChild(QtWidgets.QLabel,"label_image")
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        self.label_image.setScaledContents(True)
        self.label_image.show()
        # 宣告一個RBFN model
        self.RBFN = MyRBFN(h_layers = 20,sigma = 1)
        # 宣告一個Mycar
        self.Mycar = car()
        # 連結按鈕事件
        self.pushButton_GO = self.findChild(QtWidgets.QPushButton,"pushButton_GO")
        self.pushButton_GO.clicked.connect(self.auto_car_move)
        # Show the GUI
        self.show()

    
    def auto_car_move(self):
        self.Mycar
        self.Mycar.car_move(0)
        self.plot_new_graph(self.Mycar.x,self.Mycar.y,self.Mycar.phi)
        
    def plot_new_graph(self,x,y,phi):
        
        ax = self.fig.add_subplot(111)
        #畫起點
        plt.plot((-6,6),(0,0),c = "r")
        #畫終點線
        rect = patches.Rectangle((self.map[1][0],self.map[2][1]),self.map[2][0]-self.map[1][0],self.map[1][1]-self.map[2][1],color = 'r')#左下座標,長度,寬度
        ax.add_patch(rect)
        #畫牆壁
        temp = self.map[3:]
        for i in range(len(temp)-1): 
            plt.plot((temp[i][0],temp[i+1][0]),(temp[i][1],temp[i+1][1]),c = 'b')
        #畫車
        plt.scatter(x,y,c = 'r')
        circle = patches.Circle((x,y),radius = 3,fill = False)
        ax.add_patch(circle)
        #感測器線條
        length = 10
        endy = y + length * math.sin(math.radians(phi))
        endx = length * math.cos(math.radians(phi))
        plt.plot([x,endx],[y,endy],c = 'r')
        plt.xlim([-20,40])
        plt.ylim([-10,55])
        plt.savefig("pic")
        #更新畫面
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        plt.clf()
        


def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MyGUI() # Create an instance of our class
    app.exec_() # Start the application
    
    #刪掉圖片
    if os.path.exists("pic.png"):
        os.remove("pic.png")

if __name__ == '__main__':
    main()