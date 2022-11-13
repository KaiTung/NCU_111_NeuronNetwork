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
        self.p = Playground()
        self.p.RBFN.fit(k=40)
        # 取得label_image
        self.label_image = self.findChild(QtWidgets.QLabel,"label_image")
        self.label_image.setScaledContents(True)
        # 連結按鈕事件
        self.pushButton_GO = self.findChild(QtWidgets.QPushButton,"pushButton_GO")
        self.pushButton_GO.clicked.connect(self.run)

        self.p.draw_new_graph()
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        self.label_image.show()

        # Show the GUI
        self.show()

    def run(self):
        # use example, select random actions until gameover
        state = self.p.reset()
        # fitting RBF-Network with data
        while not self.p.done:
            # print every state and position of the car
            c =  self.p.car.getPosition('center')
            # select action randomly
            # you can predict your action according to the state here
            # action = p.predictAction(state)
            action = self.p.RBFN.predict([state])
            action *= 10
            # action = model.predict(state)
            # take action
            print("state={},center={},action={}".format(state, self.p.car.getPosition('center'),self.p.RBFN.predict([state])))
            state = self.p.step(action)
            self.p.draw_new_graph()
            self.show_new_graph()

        
    def show_new_graph(self):
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        QtWidgets.QApplication.processEvents()


def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MyGUI() # Create an instance of our class
    app.exec_() # Start the application
    
    #刪掉圖片
    if os.path.exists("pic.png"):
        os.remove("pic.png")

if __name__ == '__main__':
    main()