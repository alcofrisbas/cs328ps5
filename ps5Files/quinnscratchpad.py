import pandas as pd
import math
import matplotlib.pyplot as plt

def calc_output(x1, weights):
    """
    calculates the output of the given neural
    network using the input and two weights.
    """

    n1 = weights[0]*x1
    x2 = 1.0/(1+math.exp(n1))

    n2 = weights[1]*x2
    x3 = 1.0/(1+math.exp(n2))

    return x3

def sum_squared_error(inOuts, weights):
    SSE = 0
    for i in inOuts:
        start = i[0]
        y = i[1]
        gWx = calc_output(start, weights)
        squaredError = (y-gWx)**2
        SSE += squaredError
    return SSE


### BELOW HERE

results = []
for w1 in range(-5,6):
    row = []
    for w2 in range(-5,6):
        row.append(nn_practice.sum_squared_error(((0,1),(1,0)), (w1,w2)))
    results.append(row)


a = plt.contourf(range(-5,6),range(-5,6),results_jank)
plt.colorbar(a)
plt.title('SSE for different w1 and w2')
plt.xlabel('w1')
plt.ylabel('w2')
plt.show()
