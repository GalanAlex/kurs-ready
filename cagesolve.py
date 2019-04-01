import numpy as np
import matplotlib.pyplot as plt

import solve as slv

MODE = 'c'

J = 100
K = 100


Lx = 10
Ly = 6
L = 4
lambd = 2
n = 1
z = 1
x = 1
y = 1
c = 1
h = Ly / J
tau = L / K





o = complex(0,1)
k1 = 2*np.pi/lambd
alpha = ((o*(np.pi**2))/(4*k1*n*(Lx**2)))





def cons():
    return((np.pi**2)/(Lx**2))
def psi(j,h):
    return np.exp(-(((j*h-Ly/2)/(0.1*Ly))**2))

def mainsolve(J,K):
    V = np.zeros((J+1, K + 1), dtype=np.complex)

    if MODE == 'c':

        h = Ly / J
        tau = L / K
        s = ((np.pi**2)/(2*(Lx**2))+(1/(h**2)))
        a = (1-alpha*tau*s)
        print(a)

        b = ((alpha*tau)/(2*(h**2)))
        c = (1+alpha*tau*s)
        p = np.zeros((J,K), dtype=np.complex)

        # print(a)
        # print(b)
        beta = (tau * alpha * ((np.pi ** 2) / (Lx ** 2)))
        mu = (alpha * ((tau) / (h ** 2)))

        j = 1
        while j <= J - 1:
            V[j, 0] = psi(j, h)
            # print("j = ",j," Решение - ",V[j, 0])
            j = j + 1

        alpha1 = np.zeros((J - 1), dtype=np.complex)
        alpha1[1] = -(b / a)
        j = 2
        # print(alpha1[1])
        while j < J - 1:
            alpha1[j] = -(b / (b * alpha1[j - 1] + a))
            # print("j = ",j," Решение - ",alpha1[j])
            j = j + 1

        betta = np.zeros((J,K+1), dtype=np.complex)
        k=0
        while k<=K-1:
            p[1,k] = c*V[1,k] -b*V[2,k]
            p[J-1,k] = -b*V[J-2,k]+c*V[J-1,k]
            j = 2
            while j <= J-2:
                p[j,k] = -b*V[j-1,k] +c*V[j,k] -b*V[j+1,k]
                j=j+1
            betta[1, k] = p[1, k] / a
            j = 2
            while j <= J - 2:
                betta[j, k] = (p[j, k] - b * betta[j - 1, k]) / (b * alpha1[j - 1] + a)
                j = j + 1
            V[J - 1, k + 1] = (p[J - 1, k] - b * betta[J - 2, k]) / (b * alpha1[J - 2] + a)
            V[J-1,k+1] = V[J-1,k+1]
            j = J - 2
            while j >= 1:
                V[j, k + 1] = alpha1[j] * V[j + 1, k + 1] + betta[j, k]
                V[j,k+1] = V[j,k+1]
                j = j - 1

            k = k + 1
        '''

        k = 0
        while k < K:
            j = 1
            while j < 17:
                V[j,k] = V[j,k]*0.000001
                j = j+1
            j = J
            while j > J-17:
                V[j,k] = V[j,k]*0.000001
                j=j-1
            k = k+1
        '''
        #print(V[:,0])
        #print(p[1,0])
        #print(V[5,:])



    if MODE == 'o':
        V = np.zeros((J+1,K+1), dtype=np.complex )
        h = Ly / J
        tau = L / K
        c = (4 * (k1 ** 2) * (n ** 2) ) / 4*(1 + ((np.pi ** 2) * (Ly ** 4)) / (16 * (Lx ** 2)))
        j = 1
        while j <= J-1:
            V[j, 0] = psi(j,h)
            #print(V[j,0])
            j = j+1
        k = 0
        if(tau <= c*(h**4)):
            print("ejfbwpeifbwepif")
            while k < K:
                #V[J, k + 1] = 0
                V[1, k + 1] = ((alpha*tau*((np.pi**2)/(Lx**2))+1)*V[1,k] - ((alpha*tau)/(h**2))*(V[2,k] - 2*V[1,k]))*0.985
                j=2
                while j <= J-1:
                    V[j,k+1] = ((alpha*tau*((np.pi**2)/(Lx**2))+1)*V[j,k] - ((alpha*tau)/(h**2))*(V[j+1,k] - 2*V[j,k] - V[j-1,k]))*0.985

                    #V[j,k+1] = (abs(alpha)*tau*((np.pi**2)/(Lx**2))*V[j,k] - (abs(alpha)*tau*(V[j+1,k] - 2*V[j,k] + V[j-1, k]))/(h**2) + V[j,k])
                    #print(V[j,k+1])
                    j = j+1
                V[J - 1, k + 1] = ((alpha * tau * ((np.pi ** 2) / (Lx ** 2)) + 1) * V[J - 1, k] - (
                                (alpha * tau) / (h ** 2)) * (V[J - 2, k] - 2 * V[J - 1, k]))
                #print(V[:,k])
                k = k+1


        #print(V.shape)




    if MODE == 'im':
        h = Ly/J
        tau = L/K
        s = ((2)/(h**2))-((np.pi**2)/(Lx**2))

        a = 1 - (alpha*tau*s)
        b = (alpha*((tau)/(h**2)))
        # print(a)
        # print(b)
        beta = (tau * alpha * ((np.pi ** 2) / (Lx ** 2)))
        mu = (alpha * ((tau) / (h ** 2)))

        V = np.zeros((J+1, K + 1), dtype=np.complex)
        j = 1
        while j <= J - 1:
            V[j, 0] = psi(j, h)
            # print("j = ",j," Решение - ",V[j, 0])
            j = j + 1

        alpha1 = np.zeros((J - 1), dtype=np.complex)
        betta = np.zeros((J - 1, K + 1), dtype=np.complex)
        alpha1[1] = -(b / a)
        j = 2
        # print(alpha1[1])
        while j <= J - 2:
            alpha1[j] = -(b / (b * alpha1[j - 1] + a))
            # print("j = ",j," Решение - ",alpha1[j])
            j = j + 1
        k = 0
        while k <= K - 1:
            j = 2

            betta[1, k] = V[1, k] / a
            # print(betta[1, k])
            while j <= J - 2:
                betta[j, k] = (V[j, k] - b * betta[j - 1, k]) / (b * alpha1[j - 1] + a)
                # print("j = ",j," betta - ",betta[j,k])
                j = j + 1
            V[J - 1, k + 1] = (V[J - 1, k] - b * betta[J - 2, k]) / (b * alpha1[J - 2] + a)
            j = J - 2
            while j >= 1:
                V[j, k + 1] = alpha1[j] * V[j + 1, k + 1] + betta[j, k]
                j = j - 1
            k = k + 1
            # print(V[:,1])

    return V



if __name__ == "__main__":
    mainsolve(J,K)
    '''
    f = open('a.txt', 'w')
    l =0
    m = 0
    print(v[1,0])
    while l<J:

        while m < K:
            f.write(str(v[l,m]))
            m=m+1
        l=l+1
    f.close()
    '''