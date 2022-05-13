import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

'''
This script does Newtion interpolation and then also makes a spline function,
based on given data points.
'''

def read_data():

   lines = []
   with open('interpolation/data9.txt') as f:
       lines = f.readlines()

   x = np.zeros(len(lines))
   y = np.zeros(len(lines))

   for i in range(len(lines)):
       tmp = lines[i].split()
       x[i] = tmp[0]
       y[i] = tmp[1]

   return x, y

def find_parameters(x, y):

    n = len(x)
    diffs = []
    diffs.append(y)
    b = []
    b.append(y[0])
    for i in range(1, n):
        diffs.append(np.zeros(n-i))
        for j in range(n-i):
            diffs[i][j] = (diffs[i-1][j+1] - diffs[i-1][j])/(x[j+i]-x[j])
        b.append(diffs[i][0])
    return b

def w(x, b, xi):
    n = len(xi)
    tmp = 1.
    y = b[0]
    for i in range(n-1):
        tmp = tmp*(x - xi[i])
        y = y + tmp*b[i+1]
    return y

def newton(x, y):
    # Newton
    b = find_parameters(x, y)
    length = 200
    x1 = np.linspace(x[0],x[-1], length)
    y1 = np.zeros(length)
    for i in range(length):
        y1[i] = w(x1[i], b, x)

    return x1, y1

def spline(x, y):
    # 3-rd order spline funcion
    n = len(x)
    h = np.zeros(n-1)
    for i in range(n-1):
        h[i] = x[i+1] - x[i]
    
    A = np.zeros((n,n))
    A[0][0] = 1
    A[-1][-1] = 1
    for i in range(0,n-2):
        A[i+1][i+1] = 2*(h[i]+h[i+1])
        A[i+1][i] = h[i]
        A[i+1][i+2] = h[i+1]

    b = np.zeros((n,1))
    diffs = np.zeros(n-1)
    for i in range(n-1):
        diffs[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    for i in range(1,n-1):
        b[i]=3*(diffs[i]-diffs[i-1])

    c = np.matrix.flatten(solve(A, b))

    b = np.zeros(n-1)
    d = np.zeros(n-1)
    a = np.zeros(n-1)
    
    for i in range(n-1):
        b[i] = (y[i+1]-y[i])/h[i] - (h[i]*(2*c[i]+c[i+1]))/3
        d[i] = (c[i+1] -c[i])/(3*h[i])
        a[i] = y[i]

    s = 10 # samples per interval

    xout = np.array([])
    yout = np.array([])

    for i in range(n-1):
        xtmp = np.linspace(x[i],x[i+1],s)
        ytmp = np.zeros(s)
        for j in range(s):
            diff = xtmp[j] - x[i]
            ytmp[j] = a[i] + b[i]*diff + c[i]*(diff**2) + d[i]*(diff**3)
        xout = np.hstack((xout, xtmp))
        yout = np.hstack((yout, ytmp))
    return xout, yout


x, y = read_data()

xn, yn = newton(x,y)

xg, yg = spline(x, y)


plt.plot(xn,yn, label="Interpolacja Newtona")
plt.plot(xg,yg, label="Funkcja sklejana")
plt.plot(x,y, linestyle='', marker='o', label="Pierwotne punkty")
plt.legend()
plt.show()

