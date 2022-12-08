import numpy as np

class MyHopifield():
    def init(x):
        self.w = None
        self.x = x
        self.mode = "basic"
    
    def change_mode(self):
        if self.mode == "basic":
            self.mode = "bonus"
        else:
            self.mode = "basic"
    
    def update_P_N(self):
        if self.mode == "basic":
            self.P = 9*12
            self.N = 3
        elif self.mode == "bonus":
            self.P = 10*10
            self.N = 15
        
    def calculate_w():
        self.w = (np.dot(self.x,self.x.T) - self.N) / self.P
        

def data_process():
    path_to_file = "Basic_Training.txt"
    
    temp = []
    temp2 = []
    for line in open(path_to_file,'r'):
        
        if line != "\n":
            temp.append(line)
        else:
            temp2.append(temp)
            temp.clear()
    print(temp2)


def main():
    data_process()
    
if __name__ == "__main__":
    main()