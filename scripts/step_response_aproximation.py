import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

'''
This script uses minimal squares method to approximate second order transmitance
with damping lesser than 1 (this assumption is done to find explicit time domain solution and use that,
instead of re-runing a simulation), for given step response measurement
'''

def step_response(t, tau, zeta, tau_z, k):
    # g(t)
    f = 1/(tau**2)*(1-zeta**2)
    g = np.multiply(np.exp(-1/tau*zeta*t), np.sin(np.sqrt(f)*t))*f
    
    # h(t)
    h = np.cumsum(np.multiply(t,g))

    return k*(np.add(tau_z*g,h))

def norma(x):
    # x = [tau, zeta, tau_z, k]
    new_h = step_response(t, x[0], x[1], x[2], x[3])
    return np.sum(np.square(np.subtract(h, new_h)))

def read_data():

    lines = []
    with open('step_response_aproximation/data9.txt') as f:
        lines = f.readlines()

    t = np.zeros(len(lines))
    h = np.zeros(len(lines))

    for i in range(len(lines)):
        tmp = lines[i].split()
        t[i] = tmp[0]
        h[i] = tmp[1]

    return t, h

global t, h
t, h = read_data()

#t = np.linspace(0,15,1000)
#plt.plot(t, step_response(t, tau, zeta, tau_z, k))
x = fmin(norma, np.array([1,0.5,-100,1]))
print(x)

tau = x[0]
zeta = x[1]
tau_z = x[2]
k = x[3]

plt.plot(t, h, t, step_response(t, tau, zeta, tau_z, k))
plt.show()
