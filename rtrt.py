import numpy as np

Ly=6
J=50
y=1

def psi(j,h):
    return np.exp(-((j*h-Ly/2)/0.1*Ly)**2)


if __name__ == "__main__":
    h=y/J
    print(psi(J,h))