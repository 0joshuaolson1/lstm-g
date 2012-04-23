import math
def log(val, flg):
    if flg < 1:
        return 1. / (1 + math.exp(-val))
    evl = log(val, 0)
    return evl * (1 - evl)
def non(val, flg):
    return val * (1 - flg)
def fnx():
    return [non, log]
def lsm(fmt):
    arr=[{}, {}, {}]
    for lin in fmt.split("\n"):
        con = lin.split(" ")
        if len(con) < 3:
            arr[1][con[0]] = [0, 0, con[1], [], 0, 0, 0]
        else:
            arr[0][con[0], con[1]] = [con[2], con[3], 0]
            if con[3] > -1:
                arr[1][con[3]][3].append((con[0], con[1]))
    for j, i in arr[0]:
        for k, l in arr[1][j][3]:
            arr[2][j, i, k] = 0
    return arr
def str(arr):
    fmt = ""
    for j in arr[1]:
        fmt += "\n" + j + " " + arr[1][j][2]
    for i in arr[1]:
        for j in arr[1]:
            if (j, i) in arr[0]:
                fmt += "\n" + j + " " + i + " " + arr[0][j, i][0] + " " + arr[0][j, i][1]
    return fmt[1:]
def gan(j, i):
    if arr[0][j, i][1]<0:
        return 1
    return arr[1][arr[0][j, i][1]][1]
def fwd(ar2, dat):
    arr = ar2[:]
    for j in dat:
        arr[1][j][1] = dat[j]
    for j in range(len(dat), len(arr[1])):
        if (j, j) in arr[0]:
            arr[1][j][0] *= gan(j, j)
        else:
            arr[1][j][0] = 0
        for i in arr[1]:
            if j != i and (j, i) in arr[0]:
                arr[1][j][0] += gan(j, i) * arr[0][j, i][0] * arr[1][i][1]
        arr[1][j][1] = fnx()[arr[1][j][2]].evl(arr[1][j][0], 0)
        for i in arr[1]:
            if j != i and (j, i) in arr[0]:
                if (j, j) in arr[0]:
                    arr[0][j, i][2] *= gan(j, j)
                else:
                    arr[0][j, i][2] = 0
                arr[0][j, i][2] += gan(j, i) * arr[1][i][1]
                m = -1
                for k, l in arr[1][j][3]:
                    if m != k:
                        m = k
                        if (k, k) in arr[0]:
                            arr[2][j, i, k] *= gan(k, k) * arr[0][k, k][0]
                            if (k, k) in arr[1][j][3]:
                                arr[2][j, i, k] += fnx()[arr[1][j][2]].evl(arr[1][j][0], 1) * arr[0][j, i][2] * arr[0][k, k][0] * arr[1][k][0]
                        else:
                            arr[2][j, i, k] = 0
                        for p, q in arr[1][j][3]:
                            if p == k and q != k:
                                arr[2][j, i, k] += fnx()[arr[1][j][2]].evl(arr[1][j][0], 1) * arr[0][j, i][2] * arr[0][k, q][0] * arr[1][q][1]
    return arr
def bwd(ar2, dat, lrn):
    arr = ar2[:]
    for j in range(len(arr[1]) - len(dat), len(arr[1])):
        arr[1][j][4] = dat[j] - arr[1][j][1]
    for j in reversed(range(len(arr[1]) - len(dat))):
        arr[1][j][5] = 0
        for k in range(j + 1, len(arr[1])):
            if (k, j) in arr[0]:
                arr[1][j][5] += arr[1][k][4] * gan(k, j) * arr[0][k, j][0]
        arr[1][j][5] *= fnx[arr[1][j][2]](arr[1][j][1], 1)
        arr[1][j][6] = 0
        m = -1
        for k, l in arr[1][j][3]:
            if m != k and k > j:
                m = k
                if (k, k) in arr[0]:
                    arr[1][j][6] += arr[1][k][4] * arr[0][k, k][0] * arr[1][k][0]
                for p, q in arr[1][j][3]:
                    if p == k and q != k:
                        arr[1][j][6] += arr[1][k][4] * arr[0][k, q][0] * arr[1][q][1]
        arr[1][j][6] *= fnx[arr[1][j][2]](arr[1][j][1], 1)
        arr[1][j][4] = arr[1][j][5] + arr[1][j][6]
    for j, i in arr[0]:
        arr[0][j, i][0] += lrn * arr[1][j][5] * arr[0][j, i][2]
        m = -1
        for k, l in arr[1][j][3]:
            if m != k and k > j:
                arr[0][j, i][0] += lrn * arr[1][k][4] * arr[2][j, i, k]
    return arr
#{j,i:[w,gater,epsilon]}
#{j:[s,y,f,gated=[],sigma,sigma_p,sigma_g]}
#{j,i,k:epsilon_k}
#
#j fnx_index
#...
#j i w gater
#...