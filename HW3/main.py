import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtCore import QThread
from MyHopfidle import *
import sys
import os

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.Hopefield = MyHopifield()
        
        # 取得label_image
        self.label_img_before = self.findChild(QtWidgets.QLabel,"label_img_before")
        self.label_img_before.setScaledContents(True)
        self.label_img_after = self.findChild(QtWidgets.QLabel,"label_img_after")
        self.label_img_after.setScaledContents(True)

        self.label_img_before.setPixmap(QtGui.QPixmap("before_img_0.png"))
        self.label_img_after.setPixmap(QtGui.QPixmap("after_img_0.png"))

        #conbobox
        self.comboBox = self.findChild(QtWidgets.QComboBox,"comboBox")

        self.pushButton_OK = self.findChild(QtWidgets.QPushButton,"pushButton_OK")
        self.pushButton_OK.clicked.connect(self.train)
        
        # Show the GUI
        self.show()

    def train(self):
        return

def delete_img():
    i=0
    while 1:
        try:
            os.remove("train_img_{}.png".format(i))
            os.remove("before_img_{}.png".format(i))
            os.remove("after_img_{}.png".format(i))
        except Exception as e:
            print(e)
            break
        i+=1

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MainWindow() # Create an instance of our class
    app.exec_() # Start the application
            

if __name__ == '__main__':
    main()