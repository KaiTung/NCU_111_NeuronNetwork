import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyQt5
from PyQt5 import QtWidgets , uic
import os
import sys

class MyGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.show() # Show the GUI
        #初始化地圖
        self.map=[]
        #讀取&儲存座標點
        with open('軌道座標點.txt','r') as f:
            for i in f.read().splitlines():
                self.map.append(np.array(i.split(',')).astype(int))
        self.plot_new_graph()


    def plot_new_graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #畫終點線
        rect = patches.Rectangle((self.map[1][0],self.map[2][1]),self.map[2][0]-self.map[1][0],self.map[1][1]-self.map[2][1],color = 'r')#左下座標,長度,寬度
        ax.add_patch(rect)
        #畫牆壁
        temp = self.map[3:]
        for i in range(len(temp)-1): 
            plt.plot((temp[i][0],temp[i+1][0]),(temp[i][1],temp[i+1][1]),c = 'b')
        #畫車
        plt.scatter(self.map[0][0],self.map[0][1],c = 'r')
        plt.show()


def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MyGUI() # Create an instance of our class
    app.exec_() # Start the application

if __name__ == '__main__':
    main()