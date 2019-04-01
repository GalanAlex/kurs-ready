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
alpha = abs((o*(np.pi**2))/(4*k1*n*(Lx**2)))





def cons():
    return((np.pi**2)/(Lx**2))
def psi(j,h):
    return np.exp(-(((j*h-Ly/2)/(0.1*Ly))**2))

def mainsolve(J,K):
    V = np.zeros((J+1, K + 1))

    if MODE == 'c':

        h = Ly / J
        tau = L / K
        s = ((np.pi**2)/(2*(Lx**2))+(1/(h**2)))
        a = (1-alpha*tau*s)
        b = ((alpha*tau)/(2*(h**2)))
        c = (1+alpha*tau*s)
        p = np.zeros((J,K))

        # print(a)
        # print(b)
        beta = (tau * alpha * ((np.pi ** 2) / (Lx ** 2)))
        mu = (alpha * ((tau) / (h ** 2)))

        j = 1
        while j <= J - 1:
            V[j, 0] = psi(j, h)
            # print("j = ",j," Решение - ",V[j, 0])
            j = j + 1

        alpha1 = np.zeros((J - 1))
        alpha1[1] = -(b / a)
        j = 2
        # print(alpha1[1])
        while j < J - 1:
            alpha1[j] = -(b / (b * alpha1[j - 1] + a))
            # print("j = ",j," Решение - ",alpha1[j])
            j = j + 1

        betta = np.zeros((J,K+1))
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
                V[j,k+1] = V[j,k+1]*0.985
                j = j - 1

            k = k + 1
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

        #print(p[1,0])
        #print(V[5,:])



    if MODE == 'o':
        V = np.zeros((J+1,K+1))
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
                print(V[:,k])
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

        V = np.zeros((J+1, K + 1))
        j = 1
        while j <= J - 1:
            V[j, 0] = psi(j, h)
            # print("j = ",j," Решение - ",V[j, 0])
            j = j + 1

        alpha1 = np.zeros((J - 1))
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
            betta = np.zeros((J - 1, K + 1))
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
        '''
        h = Ly / J  # Hy
        tau = L / K  # Hz
        #mu = alpha * tau/(h**2)
        #print(mu)
        beta = tau*alpha * ((np.pi**2)/(Lx**2))
        #print(beta)
        #betta = alpha / 2

        #A = np.zeros((J , J))

        mu = alpha * tau/(h**2)
        beta = tau*alpha * ((np.pi**2)/(Lx**2))



        #a = abs(1 - tau * alpha - ((2 * betta * tau) / (h ** 2)))
        #b = abs(betta * tau / h ** 2)
        #c = abs(1 + tau * alpha + ((2 * betta * tau) / (h ** 2)))
        a = abs(1-2*mu-beta)
        b = abs(mu)
        j = 0
        while j < J - 1:
            V[j, 0] = psi(j, h)
            #print(V[j,0])
            j = j + 1

        balpha[0] = -b / a
        balpha[1] = b / (b * balpha[0] + a)
        m = 2
        while m <= J - 3:
            balpha[m] = -(b / (b * balpha[m - 1] + a))
            m = m + 1

        kk = 0
        while kk <= K - 2:
            #p[0, kk] = c * V[1, kk] - b * V[2, kk]
            p[0,kk] = V[1,kk]
            #p[J - 2, kk] = -b * V[J - 3, kk] + c * V[J - 2, kk]
            p[J-2,kk]=V[J-1,kk]
            l[0, kk] = p[0, kk] / a
            m = 1
            while m <= J - 3:
                #p[m, kk] = -b * V[m, kk] + c * V[m + 1, kk] - b * V[m + 2, kk]
                p[m,kk] = V[m+1,kk]
                l[m, kk] = (p[m, kk] - b * l[m - 1, kk]) / (b * balpha[m - 1] + a)
                m = m + 1
            V[J - 1, kk + 1] = (p[J - 2, kk] - b * l[J - 3, kk]) / (b * balpha[J - 3] + a)

            m = J - 2
            while m != 1:
                V[m, kk + 1] = balpha[m-1] * V[m + 1, kk + 1] + l[m-1, kk]
                m = m - 1
            V[1,kk+1] = balpha[0]*V[2,kk+1]+l[0,kk]
            kk = kk + 1


        
        j = 0
        while j < J-1:
            V[j, 0] = psi(j,h)
            #print(psi(j,h))
            A[j+1, j] = abs(mu)
            #print(A[j+1, j])
            A[j, j] = abs(1 - 2 * mu - beta)
            #print( A[j, j])
            A[j, j+1] = abs(mu)
            j = j+1
        print(A)

        V[0,0] = psi(0,h)
        #print(psi(J,h))
        V[J-1, 0] = psi(J, h)
        #print(psi(J, h))
        A[0,0] = abs(1 - 2 * mu - beta)
        A[0,1] = abs(mu)
        A[J-1, J-1] = abs(1 - 2 * mu - beta)
        A[J-1, J-2 ] = abs(mu)
        #print(A.shape)
        


        k = 0
        while k < K-1:
            V[:, k+1] = np.linalg.solve(A, V[:,k])
            k = k+1
        '''
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