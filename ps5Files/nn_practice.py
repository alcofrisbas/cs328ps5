import math

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

# import pandas as pd
# results = []
# for w1 in range(-5,6):
#     for w2 in range(-5,6):
#         err = {'w1':w1, 'w2':w2, SSE = sum_squared_error(((0,1),(1,0)), weights)}
#         results.append(err)

# results = pd.from_dict(results)
def compute_w2_update():
    pass