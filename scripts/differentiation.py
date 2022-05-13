import numpy as np
import matplotlib.pyplot as plt

'''
This scripts solves a differentional equation with Euler's, Heun's and middle-point method
In this case the equation is:

dy/dt = -(-1+4*t^3)*sqrt(y) where y(0) = 1

The analytical solution was found by myself on paper and is given here for comparison.
The differences are shown on a simple graph.
'''

def alnalytical(t0, tk, n):
    t = np.linspace(t0,tk,n)
    y = np.subtract(t, np.power(t,4))+2*np.sqrt(2)
    y = np.power(y/2,2)

    return y

def euler(t0, tk, n):

    h = (tk-t0)/n

    y = np.zeros(n)
    y[0] = 2
    t = np.linspace(t0,tk,n)

    for i in range(1,n):
        y[i] = y[i-1] + h*(1-4*(t[i-1]**3))*np.sqrt(y[i-1])

    return y

def heun(t0, tk, n):
    h = (tk-t0)/n

    y = np.zeros(n)
    y[0] = 2
    t = np.linspace(t0,tk,n)

    d1 = 0
    d2 = 0

    for i in range(1,n):
        d1 = (1-4*(t[i-1]**3))*np.sqrt(y[i-1])
        y0 = y[i-1] + h*d1
        d2 = (1-4*(t[i]**3))*np.sqrt(y0)

        y[i] = y[i-1] + h*(d1+d2)/2

    return y

def midpoint(t0, tk, n):
    h = (tk-t0)/n

    y = np.zeros(n)
    y[0] = 2
    t = np.linspace(t0,tk,n)

    d1 = 0
    d2 = 0

    for i in range(1,n):
        d1 = (1-4*(t[i-1]**3))*np.sqrt(y[i-1])
        y05 = y[i-1] + h*d1/2
        d2 = (1-4*((t[i-1]+h/2)**3))*np.sqrt(y05)

        y[i] = y[i-1] + h*d2

    return y


def main():
    t0 = 0
    tk = 1.25
    n = 10

    plt.plot(np.linspace(t0,tk,n),alnalytical(t0,tk,n), label="Analytical", marker=".")
    plt.plot(np.linspace(t0,tk,n),euler(t0,tk,n), label="Euler's", marker=".")
    plt.plot(np.linspace(t0,tk,n),heun(t0,tk,n), label="Heun's", marker=".")
    plt.plot(np.linspace(t0,tk,n),midpoint(t0,tk,n), label="Middle Point", marker=".")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
