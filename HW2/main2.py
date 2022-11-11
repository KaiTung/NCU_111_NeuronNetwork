import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyQt5
from PyQt5 import QtWidgets, QtGui, uic
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
        self.p.RBFN.fit(k=40)
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

    def run(self):
        # use example, select random actions until gameover
        state = self.p.reset()

        # fitting RBF-Network with data

        while not self.p.done:
            # self.plot_new_graph()
            # print every state and position of the car
            print("state={},center={},action={}".format(state, self.p.car.getPosition('center'),self.p.RBFN.predict([state])))

            # select action randomly
            # you can predict your action according to the state here
            # action = p.predictAction(state)
            action = self.p.RBFN.predict([state])
            # action = model.predict(state)
            # take action
            state = self.p.step(action)
        print("="*10,"DONE","="*10)
        
    # def plot_new_graph(self):
        
    #     ax = self.fig.add_subplot(111)
    #     #畫起點
    #     plt.plot((-6,6),(0,0),c = "r")
    #     #畫終點線
    #     rect = patches.Rectangle((self.map[1][0],self.map[2][1]),self.map[2][0]-self.map[1][0],self.map[1][1]-self.map[2][1],color = 'r')#左下座標,長度,寬度
    #     ax.add_patch(rect)
    #     #畫牆壁
    #     for l in self.lines:
    #         plt.plot((l.p1.x,l.p2.x),(l.p1.y,l.p2.y),c = 'b')
    #     #畫車
    #     plt.scatter(self.p.car.xpos,self.p.car.ypos,c = 'r')
    #     circle = patches.Circle((self.p.car.xpos,self.p.car.ypos),radius = 3,fill = False)
    #     ax.add_patch(circle)

    #     #更新畫面
    #     self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
    #     plt.clf()
        

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MyGUI() # Create an instance of our class
    app.exec_() # Start the application
    
    #刪掉圖片
    if os.path.exists("pic.png"):
        os.remove("pic.png")

if __name__ == '__main__':
    main()