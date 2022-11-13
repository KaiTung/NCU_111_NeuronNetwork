import numpy as np

def select_centers(self,x,k=3):
        z = []
        for i in range(k):
            z.append(x[i])

        done = 1
        last_z = []
        while done:

            S = []
            # frist kth x as centers
            for i in range(k):
                S.append([x[i]])

            last_z = z[:]
            #clustering
            for x_i in x:
                #caculate dis from x to z_i
                dis = []
                for z_i in z:
                    dis.append(self.euclidean_distance(z_i,x_i))
                #find min dis
                S[dis.index(min(dis))].append(x_i)
            #update z
            for i in range(len(z)):
                z[i] = sum(S[i]) / len(S[i])
            # print("z = {}".format(z))
            if self.cmp_list(last_z,z):
                done = 0
                # print("DONE")
        return z