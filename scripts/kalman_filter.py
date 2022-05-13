import numpy as np
import matplotlib.pyplot as plt

'''
This script utilizes a Kalman filter to correct readings given in a file.
It corrects a simple plane model, which considers only position and velocity.
Also predicts it's position at n+5 step at the end.
'''

def read_data():

    lines = []
    with open('kalman_filter_data/measurements9.txt') as f:
        lines = f.readlines()

    x = np.zeros(len(lines))
    y = np.zeros(len(lines))

    for i in range(len(lines)):
        tmp = lines[i].split()
        x[i] = tmp[0]
        y[i] = tmp[1]

    return x, y

x, y = read_data()

T = 1

F = np.array([[1,0,T,0],
              [0,1,0,T],
              [0,0,1,0],
              [0,0,0,1]])

G = np.array([[0,0],
              [0,0],
              [1,0],
              [0,1]])

H = np.array([[1,0,0,0],
              [0,1,0,0]])

Q = np.array([[0.002, 0   ],
              [0,    0.002]])

R = np.array([[2,0],
              [0,2]])

s = []
s.append(np.array([[x[0]],
                   [y[0]],
                   [0   ],
                   [0   ]]))
P = []
P.append(np.array([[5,0,0,0],
                   [0,5,0,0],
                   [0,0,5,0],
                   [0,0,0,5]]))

GQG = np.matmul(np.matmul(G,Q),np.transpose(G))


# finding trajectory
for n in range(len(x)-1):
    s_predict = np.matmul(F,s[n])
    P_predict = np.matmul(np.matmul(F,P[n]),np.transpose(F))
    P_predict = np.add(P_predict, GQG)

    z_predict = np.matmul(H,s_predict)

    e = np.subtract(np.array([[x[n+1]],[y[n+1]]]), z_predict)

    S = np.add(np.matmul(np.matmul(H, P_predict), np.transpose(H)), R)
    K = np.matmul(np.matmul(P_predict, np.transpose(H)), np.linalg.inv(S))

    s.append(np.add(s_predict, np.matmul(K,e)))
    P.append(np.matmul(np.subtract(np.identity(4), np.matmul(K, H)), P_predict))

# prediction n+5
predict = []
predict.append(s[-1])
for n in range(4):
    predict.append(np.matmul(F,predict[n]))

plt.plot(x,y, ls='', marker='x')
plt.plot([x[0] for x in s], [y[1] for y in s])
plt.plot([x[0] for x in predict], [y[1] for y in predict], ls='--')
plt.plot(predict[-1][0], predict[-1][1], marker='o')
plt.show()