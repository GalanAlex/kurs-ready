import numpy as np
from decimal import *

Lx = 10
Ly = 6
L = 4
lambd = 2
n = 1
z = 1
x = 1
y = 1


pi = 3.14159265
o = 20
eps = 0.0001
k = 0



def e1(v):
    return(np.exp(-((pi * v) /20 ) ** 2))

def e2(v,t,k):

    j = complex(0,1)
    return(np.exp((pi ** 2) * ((v ** 2) / (Ly ** 2) + (1 / (Lx ** 2))) * ((j*t) / (2 * k * n))))

def a1(v):
    return(np.sin((pi * v) / 2))

def a2(v,y):
    return(np.sin((pi * v * y) / Ly))

def a3(x):
    return(np.sin((pi*x)/Lx))

def cons():
    return(np.sqrt(pi)/(5))



def get_n_for_eps(eps):
    n = 1
    r = 1
    while r > eps :
        r = ((40 * pi ** (-3 / 2)) / n) * np.exp(-((pi * n) / 20) ** 2)
        n += 1
    return(n)



'''
def get_n_full(n_full,full2):
    global o
    o = n_full
    s = []
    full = []

    for i in range(o):
        k = 2 * pi / lambd
        full += e1(i) * e2(i, z,k) * a1(i) * a2(i, y) * a3(x)
        full2 *= cons()
    full[] = abs(full2)
    s = full
    o -= 1
    print(o)
    print(s)
    print(111)
    return(full)


'''

def raz(eps,n):
    f = n
    a1 = main_sum1(f)
    for i in range(f):
        a2 = main_sum1(i-1)
        if abs(a1-a2) < eps:
            return i
        else:
            i -= 1
    return i




def main_sum1(p):
    global o
    global o_last




    y = 1
    z = 1
    full = 0

    for i in range(p):
        k = 2 * pi / lambd
        full += e1(i) * e2(i, z,k) * a1(i) * a2(i, y) * a3(x)*cons()

    return(abs(full))

def main_sum(y, z):
    global o
    global o_last
    full = 0
    for i in range(o):
        k = 2 * pi / lambd
        full += e1(i) * e2(i, z,k) * a1(i) * a2(i, y) * a3(x)*cons()*3
    return(abs(full))


