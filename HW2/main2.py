import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyQt5
from PyQt5 import QtWidgets, QtGui, uic
from MyRBFN import MyRBFN
from simple_playground_copy import Car,Line2D,Point2D,Playground
import math
import sys
import os
        
class MyGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.fig = plt.figure()
        self.p = Playground()
        # 取得label_image
        self.label_image = self.findChild(QtWidgets.QLabel,"label_image")
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        self.label_image.setScaledContents(True)
        self.label_image.show()
        # 連結按鈕事件
        self.pushButton_GO = self.findChild(QtWidgets.QPushButton,"pushButton_GO")
        self.pushButton_GO.clicked.connect(self.run())
        # Show the GUI
        self.show()

    def open_file(self):
        path_to_file = "train4dAll.txt"
        data = []
        with open(path_to_file) as f:
            for i in f.readlines():
                i = i.split()
                data.append(i)
        data = np.array(data).astype(float)
        n_features = data.shape[1]
        x = data[:,:-1]
        y = data[:,n_features-1]
        return x,y

    def run(self):
        # use example, select random actions until gameover
        state = self.p.reset()

        x,y = self.open_file()
        # fitting RBF-Network with data
        model = MyRBFN(hidden_shape=10, sigma=1.)
        model.fit(x, y)

        while not self.p.done:
            self.plot_new_graph()
            # print every state and position of the car
            # print(state, p.car.getPosition('center'))

            # select action randomly
            # you can predict your action according to the state here
            # action = p.predictAction(state)
            action = model.predict([state])
            # action = model.predict(state)
            # take action
            state = self.p.step(action)
        print("="*10,"DONE","="*10)
        
    def plot_new_graph(self):
        
        ax = self.fig.add_subplot(111)
        #畫起點
        plt.plot((-6,6),(0,0),c = "r")
        #畫終點線
        rect = patches.Rectangle((self.map[1][0],self.map[2][1]),self.map[2][0]-self.map[1][0],self.map[1][1]-self.map[2][1],color = 'r')#左下座標,長度,寬度
        ax.add_patch(rect)
        #畫牆壁
        for l in self.lines:
            plt.plot((l.x1,l.x2),(l.y1,l.y2),c = 'b')
        #畫車
        plt.scatter(self.p.car.xpos,self.p.car.ypos,c = 'r')
        circle = patches.Circle((self.p.car.xpos,self.p.car.ypos),radius = 3,fill = False)
        ax.add_patch(circle)

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