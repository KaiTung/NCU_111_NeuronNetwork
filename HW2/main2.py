import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, uic
from simple_playground_new import Playground
from MyRBFN import *
import sys
import os

#Hyper paramater
LAYERS = 60
SIGMA = 2
K = 60
PATH_TO_FILE = "train4dAll.txt"

class MyGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.p = Playground()
        self.RBFN = MyRBFN(hidden_shape = LAYERS,sigma = SIGMA,k = K)
        self.RBFN.read_training_data(PATH_TO_FILE)
        self.RBFN.fit()
        # 取得label_image
        self.label_image = self.findChild(QtWidgets.QLabel,"label_image")
        self.label_image.setScaledContents(True)
        # 取得label_sensor
        self.label_sensor = self.findChild(QtWidgets.QLabel,"label_sensor")

        # 連結按鈕事件
        self.pushButton_GO = self.findChild(QtWidgets.QPushButton,"pushButton_GO")
        self.pushButton_GO.clicked.connect(self.run)

        self.p.draw_new_graph(init=0)
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        self.label_image.show()

        # Show the GUI
        self.show()

    def run(self):
        # use example, select random actions until gameover
        path_record_6D = ""
        path_record_4D = ""
        state = self.p.reset()
        self.p.draw_new_graph()
        self.show_new_graph()
        plt.pause(0.5)
        # fitting RBF-Network with data
        while not self.p.done:
            action = self.RBFN.predict([state])[0]
            # print("state={},center={},action={}".format(state, self.p.car.getPosition('center'),action))
            state = self.p.step(action)
            self.p.draw_new_graph()
            self.show_new_graph()
            self.label_sensor.setText("感測器距離(取至後三位): 前{} 右{} 左{}".format(round(state[0],3),round(state[1],3),round(state[2],3)))
            path_record_6D += "{} {} {} {} {} {}\n".format(round(self.p.car.getPosition().x,7),round(self.p.car.getPosition().y,7),round(state[0],7),round(state[1],7),round(state[2],7),round(action,7))
            path_record_4D += "{} {} {} {}\n".format(round(state[0],7),round(state[1],7),round(state[2],7),round(action,7))
            plt.pause(0.05)
        print("===DONE===")

        #寫入路徑紀錄
        with open("track6D.txt",'w') as f:
            f.write(path_record_6D)
        with open("track4D.txt",'w') as f:
            f.write(path_record_4D)
        
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