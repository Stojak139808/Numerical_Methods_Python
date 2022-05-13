#   z^2+z+1 / z^3 - 2.8z^2 + 1.75z - 0.3
#   
#   postać kanoniczna sterowalna:
#
#     | 0     1       0   |        | 0 |
# A = | 0     0       1   |    B = | 0 |
#     | 0.3  -1.75    2.8 |        | 1 |
#  
# C = | 1  1  1 |
#
#   c1 i c2 to macierze wagi do minimalizacji,
#   c1 odpowiada za wagę poszczególnych zmiennych stanu,
#   c2 odpowiada za wagę sygnału sterującego 
#

import numpy as np
import matplotlib.pyplot as plt

def step(A,B,C,D,n):

    # find step response for given state-space system
    xtmp = np.zeros(B.shape)
    x = np.zeros(B.shape)
    y = np.zeros(n)

    for i in range(n):
        xtmp = np.add(np.matmul(A,x),B)
        y[i] = np.add(np.matmul(C,x),D)
        x = xtmp

    return y

def step2(A,B,C,D,n,F):

    # find step response for given state-space system
    # with given LQR regulator
    xtmp = np.zeros(B.shape)
    x = np.zeros(B.shape)
    y = np.zeros(n)
    u = np.ones(n)

    for i in range(n):
        xtmp = np.add(np.matmul(A,x),B)
        y[i] = np.add(np.matmul(C,x),D)
        u[i] = u[i] - np.matmul(F,x)
        x = xtmp
        

    return y, u

def find_F(A,B,Q,R):
    
    # find F vector for given system and weights

    P = np.zeros(A.shape)
    lastP = np.ones(A.shape)
    while(np.amax(abs(P-lastP)) > 0.0001):
        lastP = P
        PB = np.matmul(P,B)
        Ptmp = np.matmul(np.transpose(B),PB) + R
        Ptmp = np.matmul(PB,np.linalg.inv(Ptmp))
        Ptmp = np.matmul(np.matmul(Ptmp, np.transpose(B)), P)
        Ptmp = P - Ptmp
        P = Q + np.matmul(np.transpose(A),np.matmul(Ptmp,A))

    F = R + np.matmul(np.transpose(B),PB)
    F = np.matmul(np.linalg.inv(F),np.transpose(B))
    F = np.matmul(np.matmul(F,P),A)

    return F

A = np.array([
    [0  ,  1,      0],
    [0  ,  0,      1],
    [0.3, -1.75, 2.8]
])

B = np.array([
    [0],
    [0],
    [1],
])

C = np.array([
    [1, 1, 1]
])

D = np.array([
    [0]
])

n = 40

y = step(A,B,C,D,n)

c1 = 1
c2 = 5

Q = np.zeros(A.shape)
for i in range(Q.shape[0]):
    Q[i][i] = c1

R = np.array([
    [c2]
])

F = find_F(A,B,Q,R)

newA = A - np.matmul(B,F)

print(newA)

newy, newu = step2(newA,B,C,D,n,F)

fig, axs = plt.subplots(2)
axs[0].plot(np.array(range(n)), y, 'ro')
axs[1].plot(np.array(range(n)), newy, 'bo', np.array(range(n)), newu, 'ro')
plt.show()

