import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtCore import QThread
from simple_playground_new import Playground
from MyRBFN import *
import sys
import os

class Thread(QThread):

  def __init__(self,RBFN,pushButton_GO,pushButton_Train,label_layers):
    super(Thread, self).__init__()
    self.RBFN = RBFN
    self.pushButton_GO = pushButton_GO
    self.pushButton_Train = pushButton_Train
    self.label_layers = label_layers

  def run(self):
    try:
        self.RBFN.fit()
        self.label_layers.setText("隱藏層數量(預設60)")
    except Exception as e:
        print(e)
        self.label_layers.setText("隱藏層數量(預設60)(錯誤:隱藏層數量請>=K)")
    self.pushButton_GO.setEnabled(True)
    self.pushButton_Train.setEnabled(True)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = uic.loadUi('MyGUI.ui', self)
        self.setWindowIcon(QtGui.QIcon("icon.png"))

        # 取得label_image
        self.label_image = self.findChild(QtWidgets.QLabel,"label_image")
        self.label_image.setScaledContents(True)
        # 取得label_sensor
        self.label_sensor = self.findChild(QtWidgets.QLabel,"label_sensor")

        self.label_layers = self.findChild(QtWidgets.QLabel,"label_layers")

        # 取得參數設定lineEdit
        self.lineEdit_layers = self.findChild(QtWidgets.QLineEdit,"lineEdit_layers")
        self.lineEdit_k = self.findChild(QtWidgets.QLineEdit,"lineEdit_k")
        self.lineEdit_sigma = self.findChild(QtWidgets.QLineEdit,"lineEdit_sigma")
        self.comboBox_file = self.findChild(QtWidgets.QComboBox,"comboBox_file")

        self.lineEdit_layers.textChanged.connect(self.lineEdit_change)
        self.lineEdit_k.textChanged.connect(self.lineEdit_change)
        self.lineEdit_sigma.textChanged.connect(self.lineEdit_change)

        #取得 slider
        self.horizontalSlider_layers = self.findChild(QtWidgets.QAbstractSlider,"horizontalSlider_layers")
        self.horizontalSlider_k = self.findChild(QtWidgets.QAbstractSlider,"horizontalSlider_k")
        self.horizontalSlider_sigma = self.findChild(QtWidgets.QAbstractSlider,"horizontalSlider_sigma")

        self.horizontalSlider_layers.valueChanged.connect(self.slider_change)
        self.horizontalSlider_k.valueChanged.connect(self.slider_change)
        self.horizontalSlider_sigma.valueChanged.connect(self.slider_change)

        #取得 checkBox_trace
        self.checkBox_trace = self.findChild(QtWidgets.QCheckBox,"checkBox_trace")

        # 連結按鈕事件
        self.pushButton_GO = self.findChild(QtWidgets.QPushButton,"pushButton_GO")
        self.pushButton_GO.clicked.connect(self.run)

        self.pushButton_Train = self.findChild(QtWidgets.QPushButton,"pushButton_Train")
        self.pushButton_Train.clicked.connect(self.model_fit)


        self.p = Playground()
        self.RBFN = MyRBFN()
        self.model_fit()

        self.p.draw_new_graph(init=0)
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        self.label_image.show()

        # Show the GUI
        self.show()

    def slider_change(self):
        self.lineEdit_layers.setText(str(self.horizontalSlider_layers.value()))
        self.lineEdit_k.setText(str(self.horizontalSlider_k.value()))
        self.lineEdit_sigma.setText(str(self.horizontalSlider_sigma.value()))

    def lineEdit_change(self):
        if self.lineEdit_layers.text()!="" and self.lineEdit_k.text()!="" and self.lineEdit_sigma.text()!="":
            self.horizontalSlider_layers.setValue(int(self.lineEdit_layers.text()))
            self.horizontalSlider_k.setValue(int(self.lineEdit_k.text()))
            self.horizontalSlider_sigma.setValue(int(self.lineEdit_sigma.text()))

    def model_fit(self):
        #更改參數並重新訓練
        try:
            LAYERS = int(self.lineEdit_layers.text())
            SIGMA = int(self.lineEdit_sigma.text())
            K = int(self.lineEdit_k.text())
            PATH_TO_FILE = self.comboBox_file.currentText()
            print("layers={} k={} sigma={} file={}".format(LAYERS,K,SIGMA,PATH_TO_FILE))
            self.RBFN.set_parameter(h = LAYERS,s = SIGMA,k = K)
            self.RBFN.read_training_data(PATH_TO_FILE)

            self.pushButton_GO.setEnabled(False)
            self.pushButton_Train.setEnabled(False)
            self.thread = Thread(self.RBFN,self.pushButton_GO,self.pushButton_Train,self.label_layers)
            self.thread.start()
        except Exception as e:
            print(e)

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
            if self.comboBox_file.currentText() == "train4dAll.txt":
                action = self.RBFN.predict([state])[0]
                path_record_4D += "{:<} {:<} {:<} {:<}\n".format(round(state[0],7),round(state[1],7),round(state[2],7),round(action,7))
            elif self.comboBox_file.currentText() == "train6dAll.txt":
                c = self.p.car.getPosition('center')
                state_6d = np.concatenate((np.array([c.x,c.y]),np.array(state)),axis=0)
                action = self.RBFN.predict([state_6d])[0]
                path_record_6D += "{:<} {:<} {:<} {:<} {:<} {:<}\n".format(round(self.p.car.getPosition().x,7),round(self.p.car.getPosition().y,7),
                                                                            round(state[0],7),round(state[1],7),round(state[2],7),round(action,7))
            # print("state={},center={},action={}".format(state, self.p.car.getPosition('center'),action))
            state = self.p.step(action)
            self.p.draw_new_graph(trace = self.checkBox_trace.isChecked())
            self.show_new_graph()
            text = "感測器距離(取至後三位): 前{:<5} 右{:<5} 左{:<5}".format(round(state[0],3),round(state[1],3),round(state[2],3))
            self.label_sensor.setText(text)
            plt.pause(0.02)
        print("===DONE===")

        #寫入路徑紀錄
        if self.comboBox_file.currentText() == "train6dAll.txt":
            with open("track6D.txt",'w') as f:
                f.write(path_record_6D)
        elif self.comboBox_file.currentText() == "train4dAll.txt":
            with open("track4D.txt",'w') as f:
                f.write(path_record_4D)
        
    def show_new_graph(self):
        self.label_image.setPixmap(QtGui.QPixmap("pic.png"))
        QtWidgets.QApplication.processEvents()



def main():
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MainWindow() # Create an instance of our class
    app.exec_() # Start the application
    
    #刪掉圖片
    if os.path.exists("pic.png"):
        os.remove("pic.png")

if __name__ == '__main__':
    main()