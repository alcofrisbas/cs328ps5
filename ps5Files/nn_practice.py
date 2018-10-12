import math

def g(n):
    return(1.0/(1+math.exp(-n)))

def gprime(n):
    return(g(n)*(1-g(n)))


def calc_output(x1, weights):
    """
    calculates the output of the given neural
    network using the input and two weights.
    """
    
    n1 = weights[0]*x1
    x2 = g(n1)
    n2 = weights[1]*x2
    x3 = g(n2)

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

def compute_w2_update(IN, target, weights, rate):
    x2 = g(weights[0]*IN)
    error = target - calc_output(IN, weights)
    deriv = gprime(x2*weights[1])
    delta = error * deriv
    return rate*delta*x2

def compute_w1_update(IN, target, weights, rate):
    pass