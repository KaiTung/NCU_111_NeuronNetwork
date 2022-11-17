

def cmp_list(l1,l2):
    for i in range(len(l1)):
        for j in range(len(l1[i])):
            if l1[i][j] != l2[i][j]:return False
    return True

def euclidean_distance(x1,x2):
    distance = 0
    for dim in range(len(x1)):
        distance += pow(x1[dim] - x2[dim],2)
    distance = pow(distance,0.5)
    return distance

def select_centers(x,k):
    count = 0
    z = []
    for i in range(k):
        z.append(x[i])

    done = 1
    last_z = []

    while done:

        if count > 300:
            break
        
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
                dis.append(euclidean_distance(z_i,x_i))
            #find min dis
            S[dis.index(min(dis))].append(x_i)
        #update z
        for i in range(len(z)):
            z[i] = sum(S[i]) / len(S[i])
        # print("z = {}".format(z))
        if cmp_list(last_z,z):
            done = 0
        count += 1

    return z