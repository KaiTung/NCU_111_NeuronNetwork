import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import PyQt5
from PyQt5 import QtWidgets , uic
import os
import sys

class MyGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.show() # Show the GUI
        
def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MyGUI() # Create an instance of our class
    app.exec_() # Start the application

if __name__ == '__main__':
    main()