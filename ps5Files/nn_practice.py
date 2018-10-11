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