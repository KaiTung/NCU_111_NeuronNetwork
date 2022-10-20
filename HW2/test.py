
import numpy as np
from numpy import random

def main():
    with open('軌道座標點.txt','r') as f:
        list = []
        for i in f.read().splitlines():
            list.append(i.split(','))
        print(list)

    for i in list:
        for j in i:
            print(j)

if __name__ == '__main__':
    main()