import numpy as np
import matplotlib.pyplot as plt

y=1
z=1
J=50
K=50
Ly = 6
Lx =2
L =4
V = np.zeros((J+1, K+1))
n =10
o = complex(0,1)
lambd = 2
def l(j,h):
    return np.exp(-((j * h - Ly / 2) / 0.1 * Ly) ** 2)
def c(J,K):
    h = y/J
    tau = z/K

    j = 0
    while j <= J:
        V[j, 0] = l(j,h)
        j = j+1
        k = 0

    while k < K:
    #V[0, k+1] = 0
    #V[J, k + 1] = 0
        #print(i, "i1")
        #print(k, "k1")
        j = 1
        while j < J:
            #print(i, "i2")
            #print(k, "k2")
            vol = 2 * np.pi / lambd
            alpha = o / 2 * vol * n
            V[j,k+1] = abs(tau*((np.pi**2)/(Lx**2))*V[j,k] - tau*(V[j+1,k] - 2*V[j,k] + V[j-1, k])/(h**2) + V[j,k])
            print(V[j,k+1])
            j = j+1
        k = k+1

    return V

if __name__ == "__main__":
    print(c(J,K))