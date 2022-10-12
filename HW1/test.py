
import os
p0 = "./NN_HW1_DataSet/"
p1 = os.listdir(p0)
print(p1)
for i in p1:
    p2 = os.listdir(p0+i)
    print(p2)
    print(".")


