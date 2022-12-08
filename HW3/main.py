import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtCore import QThread
from MyHopfidle import *
import sys
import os

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = uic.loadUi('myGUI.ui', self)
                
        # Show the GUI
        self.show()

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MainWindow() # Create an instance of our class
    app.exec_() # Start the application


if __name__ == '__main__':
    main()