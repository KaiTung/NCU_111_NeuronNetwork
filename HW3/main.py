import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtCore import QThread
from MyHopfidle import *
import sys
import os

class Thread(QThread):

  def __init__(self,Hopfield,comboBox,comboBox_2,pushButton_OK,label_img_before,label_img_after):
    super(Thread, self).__init__()
    self.Hopfield = Hopfield
    self.comboBox = comboBox
    self.comboBox_2 = comboBox_2
    self.label_img_before = label_img_before
    self.label_img_after = label_img_after
    self.pushButton_OK = pushButton_OK
    self.label_img_before.setPixmap(QtGui.QPixmap("loading.png"))
    self.label_img_after.setPixmap(QtGui.QPixmap("loading.png"))

  def run(self):
    self.Hopfield.fit(mode = self.comboBox.currentText())
    mode = self.comboBox.currentText()
    num = self.comboBox_2.currentText()
    self.label_img_before.setPixmap(QtGui.QPixmap("{}_before_{}.png".format(mode,num)))
    self.label_img_after.setPixmap(QtGui.QPixmap("{}_after_{}.png".format(mode,num)))
    self.pushButton_OK.setEnabled(True)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.Hopfield = MyHopifield()

        self.check()
        self.Hopfield.fit(mode = "Basic")
        
        # 取得label_image
        self.label_arrow = self.findChild(QtWidgets.QLabel,"label")
        self.label_arrow.setScaledContents(True)
        self.label_img_before = self.findChild(QtWidgets.QLabel,"label_img_before")
        self.label_img_before.setScaledContents(True)
        self.label_img_after = self.findChild(QtWidgets.QLabel,"label_img_after")
        self.label_img_after.setScaledContents(True)

        self.label_arrow.setPixmap(QtGui.QPixmap("arrow.png"))
        self.label_img_before.setPixmap(QtGui.QPixmap("Basic_before_0.png"))
        self.label_img_after.setPixmap(QtGui.QPixmap("Basic_after_0.png"))

        #combobox
        self.comboBox = self.findChild(QtWidgets.QComboBox,"comboBox")
        self.comboBox_2 = self.findChild(QtWidgets.QComboBox,"comboBox_2")
        self.comboBox_2.addItems(['0','1','2'])
        self.comboBox_2.currentTextChanged.connect(self.change_img)

        self.pushButton_OK = self.findChild(QtWidgets.QPushButton,"pushButton_OK")
        self.pushButton_OK.clicked.connect(self.comboBox_change)
        
        # Show the GUI
        self.show()

    def check(self):
        if os.path.exists("Basic_after_0.png"):
            self.Hopfield.basic_done = 1
        else:
            self.Hopfield.basic_done = 0

        if os.path.exists("Bonus_after_0.png"):
            self.Hopfield.bonus_done = 1
        else:
            self.Hopfield.bonus_done = 0

    def change_img(self):
        mode = self.comboBox.currentText()
        num = self.comboBox_2.currentText()
        self.label_img_before.setPixmap(QtGui.QPixmap("{}_before_{}.png".format(mode,num)))
        self.label_img_after.setPixmap(QtGui.QPixmap("{}_after_{}.png".format(mode,num)))

    def comboBox_change(self):
        self.comboBox_2.clear()
        if self.comboBox.currentText() == "Basic":
            self.comboBox_2.addItems(['0','1','2'])
        else:
            self.comboBox_2.addItems(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'])
        
        self.change_img()
        self.check()
        self.pushButton_OK.setEnabled(False)
        self.thread = Thread(self.Hopfield,self.comboBox,self.comboBox_2,self.pushButton_OK,self.label_img_before,self.label_img_after)
        self.thread.start()

def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MainWindow() # Create an instance of our class
    app.exec_() # Start the application
            
if __name__ == '__main__':
    main()