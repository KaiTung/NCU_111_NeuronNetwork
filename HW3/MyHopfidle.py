import numpy as np
import matplotlib.pyplot as plt
import os

class MyHopifield():
    def init(self):
        self.w = None

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
        self.theta = np.dot(self.w,[1]*self.P)

    def cmp(self,v1,v2):
        dif = 0
        for i in range(len(v1)):
            if v1[i] != v2[i]:
                dif+=1
        if dif > 0:
            return False
        else:
            return True

    def think(self,x):
        c = 0
        sign_ans = x
        while 1:
            if c >= 10:
                break
            temp = sign_ans

            ans = np.dot(x,self.w) - self.theta
            sign_ans = np.sign(ans)

            for i in range(len(sign_ans)):
                if sign_ans[i] == 0:
                    sign_ans[i] = x[i]
            
            if self.cmp(sign_ans,temp):
                break
            
            c+=1
        return np.array(sign_ans)

    def save_img(self,x,type):
        for i in range(x.shape[0]):
            plt.axis('off')
            plt.margins(0)
            plt.subplots_adjust(top=1,bottom=1,right=1,left=1,hspace=1,wspace=1)
            plt.imshow(x[i],cmap='Greys')
            if type == "train":
                plt.savefig("train_img_{}.png".format(i))
            if type == "before":
                plt.savefig("before_img_{}.png".format(i))
            if type == "after":
                plt.savefig("after_img_{}.png".format(i))
            


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
    hopifield = MyHopifield()

    x = data_preprocess("Basic_Training.txt")
    # x = np.array([[1,-1,1],[-1,1,-1]])
    hopifield.read_data(x)

    hopifield.calculate_w()

    test_x= []
    x2 = data_preprocess("Basic_Testing.txt")
    for xx in x2:
        test_x.append(xx.reshape(hopifield.W,hopifield.H))
    hopifield.save_img(np.array(test_x),type="before")

    think_x2=[]
    for xx in x2:
        output = hopifield.think(xx).reshape(hopifield.W,hopifield.H)
        think_x2.append(output)
    
    hopifield.save_img(np.array(think_x2),type="after")
        

if __name__ == "__main__":
    main()