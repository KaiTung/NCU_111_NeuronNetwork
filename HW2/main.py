import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyQt5
from PyQt5 import QtWidgets, QtGui, uic
import math
import sys
import os

class RBFN():
    
    def rbf(x, centers, variance):
        return np.exp(-np.linalg.norm(centers - x)**2)

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
        self.plot_new_graph(0,0,90)
        # 取得label_image
        self.label_image = self.findChild(QtWidgets.QLabel,"label_image")
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        self.label_image.setScaledContents(True)
        self.label_image.show()
        # Show the GUI
        self.show()

    
    def formula(x,y,theta,phi):
        
        new_x = x + math.cos(phi + theta) + math.sin(theta) * math.sin(phi)
        new_y = y + math.sin(phi + theta) - math.sin(theta) * math.cos(phi)
        new_phi = phi - math.asin(2*math.sin(theta)/3)
        
        return new_x,new_y,new_phi
        
    def plot_new_graph(self,x,y,degree):
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
        degree -= 45
        for i in range(0,3):
            endy = y + length * math.sin(math.radians(degree))
            endx = length * math.cos(math.radians(degree))
            plt.plot((x,endx),(y,endy),c='r')
            degree += 45
        plt.xlim([-20,40])
        plt.ylim([-10,55])
        plt.savefig("pic")
        


def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MyGUI() # Create an instance of our class
    app.exec_() # Start the application
    
    #刪掉圖片
    if os.path.exists("pic.png"):
        os.remove("pic.png")

if __name__ == '__main__':
    main()