import numpy as np
import matplotlib.pyplot as plt
import os

class MyHopifield():
    def __init__(self):
        self.w = None
        self.basic_done = 0
        self.bonus_done = 0

    def read_data(self,x):
        self.x = x
        self.N = self.x.shape[0]
        self.P = self.x.shape[1]
        if self.P == 100:
            self.H = 10
            self.W = 10
        if self.P == 108:
            self.H = 9
            self.W = 12
        
    def calculate_w(self):
        self.w = (np.dot(self.x.T,self.x) - self.N*np.identity(self.P)) / self.P
        # self.theta = np.dot(self.w,[1]*self.P)
        self.theta = 0

    def cmp(self,v1,v2):
        for i in range(len(v1)):
            if v1[i] != v2[i]:
                return False
        return True

    def think(self,x):
        x_n = x
        while 1:

            temp = np.dot(x_n,self.w) - self.theta
            x_n_plus_1 = np.sign(temp) #大於0換成1 小於0換成-1 等於0維持0

            for i in range(self.P):  #x(n+1)維持0的地方要等於x(n)
                if x_n_plus_1[i] == 0:
                    x_n_plus_1[i] = x_n[i] #x(n+1) = x(n)
            
            if self.cmp(x_n_plus_1,x_n): #一模一樣就停止
                break
            
            x_n = x_n_plus_1[:]

        return np.array(x_n_plus_1)

    def save_img(self,x,name):
        for i in range(x.shape[0]):
            plt.axis('off')
            plt.imshow(x[i],cmap='Greys')
            plt.savefig("{}_{}".format(name,i))

    def fit(self,mode):

        path_train = "Basic_Training.txt"
        path_test = "Basic_Testing.txt"

        if mode == "Bonus":
            path_train = "Bonus_Training.txt"
            path_test = "Bonus_Testing.txt"

        x = data_preprocess(path_train)

        if ( mode == "Basic" and self.basic_done != 1) or ( mode == "Bonus" and self.bonus_done != 1):
            self.read_data(x)
            self.calculate_w()

            test_x= []
            x2 = data_preprocess(path_test)
            for xx in x2:
                test_x.append(xx.reshape(self.W,self.H))
            self.save_img(np.array(test_x),name="{}_before".format(mode))

            think_x2=[]
            for xx in x2:
                output = self.think(xx).reshape(self.W,self.H)
                think_x2.append(output)
            
            self.save_img(np.array(think_x2),name="{}_after".format(mode))

            if mode == "Basic":
                self.basic_done = 1
            if mode == "Bonus":
                self.bonus_done = 1

def data_preprocess(path_to_file):    
    temp = []
    temp2 = []
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            re_line = line.replace('\n','') #換行處理掉
            if re_line != '':
                for num in re_line:
                    if num == ' ':
                        temp.append(-1) #空白補-1
                    else:
                        temp.append(int(num)) 
            else:
                temp2.append(temp[:])
                temp.clear()
    temp2.append(temp[:])
    return np.array(temp2)

def delete_img(self,mode):
    i = 0
    while 1:
        try:
            os.remove("{}_before_img_{}.png".format(mode,i))
            os.remove("{}_after_img_{}.png".format(mode,i))
        except:
            break
        i+=1

def main():
    Hopfield = MyHopifield()

    # Hopfield.fit(mode = "Basic")
    Hopfield.fit(mode = "Bonus")
        

if __name__ == "__main__":
    main()