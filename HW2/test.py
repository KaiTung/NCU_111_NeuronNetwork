
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    for i in range(5):
        plt.xlim([-15,55])
        plt.ylim([-15,55])
        # plt.show()
        plt.savefig("pic")

if __name__ == '__main__':
    main()