import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyQt5
from PyQt5 import QtWidgets, QtGui, uic
from simple_playground_copy import Car,Line2D,Point2D,Playground
from MyRBFN import *
import sys
import os

class MyGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.p = Playground()
        self.RBFN = MyRBFN(hidden_shape = 50,sigma = 2,k = 40)
        self.RBFN.fit()
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
        self.p.draw_new_graph()
        self.show_new_graph()
        plt.pause(1)
        # fitting RBF-Network with data
        while not self.p.done:
            # action = self.RBFN.predict([state])
            action = 0
            print("state={},center={},action={}".format(state, self.p.car.getPosition('center'),action))
            state = self.p.step(action)
            self.p.draw_new_graph()
            self.show_new_graph()
            plt.pause(1)
        print("===DONE===")
        
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