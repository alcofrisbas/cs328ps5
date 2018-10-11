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

# following parameters: (1) the input to the network, (2) the target (desired) output, (3) the weights as a tuple (w1,w2)(w1,w2), and (4) the learning rate η. It should return the change that should be made to w2 given that this input/output error was just observed. Test your function in the cell below, comparing the output to one that you calculate by hand.

# g′(n)=g(n)⋅(1−g(n)) 


#############
# REALLY NOT SURE ABOUT THIS....
# NEED TO TAKE A GOOD LOOK...
################
def compute_w2_update(IN, target, weights, rate):
    y_gWx = target - calc_output(IN, weights)
    input_i = 1.0/(1+math.exp(weights[0]*IN))
    
#     g = 1.0/(1+math.exp())
    
    error = target - 1.0/(1+math.exp(input_i*weights[1]))
    deriv = 1.0/(1+math.exp(input_i*weights[1])) * (1 - (1.0/(1+math.exp(input_i*weights[1]))))
    #delta_w_2 = rate * y_gWx * 
    return rate*error*deriv*input_i