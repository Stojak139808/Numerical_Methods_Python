import numpy as np
import matplotlib.pyplot as plt

'''
This scripts implements Romberg's and Gaussian quadrature (3 points in this case)
integration. Also calculates analytical integral for given polynomial.
The results are displayed in a small table.

It was really surprising (even for a simple polynomial) how accurate the quadrature was 
considrering how little calculations were done.
'''

def w(x, coeffs):
    y = 0
    val = 1
    for a in coeffs[::-1]:
        y = y + val*a
        val = val*x
    return y

def print_polynomial(coeffs):
    n = coeffs.shape[0]
    for i in range(n):
        if i == 0:
            print(coeffs[i], end="")
        elif(coeffs[i] >= 0):
            print("+ {}".format(coeffs[i]), end="")
        else:
            print("- {}".format(-1*coeffs[i]), end="")
        if n-i > 2:
            print("x^{} ".format(n-i-1), end="")
        elif n-i == 2:
            print("x ", end="")
    print()

def analytic_integral(coeffs, C, start, end):
    n = coeffs.shape[0]
    y = np.zeros(n+1)
    for i in range(n):
        y[i] = coeffs[i]/(n - i)
    y[-1] = C

    print("F(x) = ", end="")
    print_polynomial(y)
    
    return w(end, y) - w(start, y)

def trapezoidal_integral(coeffs, start, end, h):
    
    integral = 0
    
    x = start + h

    # There was a numerical error because of how float works,
    # sometimes instead fe 23, there was 22.9999999997 and the loop
    # did one too many iterations
    # that's why that -0.000000001 is here

    while x < end - 0.000000001: 
        integral = integral + w(x, coeffs)
        x = x + h
    integral = w(start, coeffs) + 2 * integral + w(end, coeffs)

    return (h/2)*integral

def romberg_integral(coeffs, start, end, h):
    n = 4

    I = np.zeros((n,n))
    for i in range(n):
        I[i,0] = trapezoidal_integral(coeffs, start, end, h)
        h = h/2

    for k in range(1, n):
        for j in range(n-k):
            I[j,k] = ((4**k)*I[j+1,k-1]-I[j,k-1])/((4**k)-1)
    print(I)
    
    return I[0,-1]

def gauss_integral(coeffs, start, end):
    integral = 0
    c0 = 5/9
    c1 = 8/9
    t0 = ((end + start) + (end - start)*(-np.sqrt(3/5)))/2
    t1 = ((end + start) + (end - start)*(0))/2
    t2 = ((end + start) + (end - start)*(np.sqrt(3/5)))/2
    integral = c0*w(t0,coeffs) + c1*w(t1,coeffs) + c0*w(t2,coeffs)

    return integral*(end-start)/2


polynomial = np.array([-0.03654, 0.7655, 0.543, 2.663, 1.543])
start = -8
end = 23
#polynomial = np.array([400, -900, 675, -200, 25, 0.2])
#start = 0
#end = 0.8
print("f(x) = ", end="")
print_polynomial(polynomial)

true_integral = analytic_integral(polynomial, 0, start, end)
print()
rom_integral = romberg_integral(polynomial, start, end, 1)
print()
gau_integral = gauss_integral(polynomial, start, end)

Table = []
Table.append(["Method", "value", "error"])
Table.append(["Analytical", true_integral, "-----"])
Table.append(["Romberg", rom_integral, 100*abs(true_integral-rom_integral)/true_integral])
Table.append(["Gauss", gau_integral, 100*abs(true_integral-gau_integral)/true_integral])

for row in Table:
    for collumn in row:
        print(" {:<24} |".format(collumn), end="")
    print()
        
        
