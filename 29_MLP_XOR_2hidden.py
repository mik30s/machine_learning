'''Mircea Agapie, April 14, 2018

   MLP with 2 inputs, 2 hidden nodes and 1 output
   Dataset in the array andy is XOR
   Weights are initialized randomly
   Option to initialize biases randomly or with zeroes
   Option to allow biases to adapt, using the same delta rule
   Activation function is sigmoid (logistic)
   Hidden nodes have their output squashed, but squashing
the output of the output node is optional
   Main loop breaks when a correct classification is reached
'''

import numpy as np
from math import exp
from random import uniform, randint

andy = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]])
lim = 2.0
w1   = uniform(-lim, lim)
w2   = uniform(-lim, lim)
w11  = uniform(-lim, lim)
w12  = uniform(-lim, lim)
w21  = uniform(-lim, lim)
w22  = uniform(-lim, lim)
bout, b1, b2 = 0.0, 0.0, 0.0
#Use these instead for non-zero biases:
'''bout = uniform(-lim, lim)           #bias for output node
b1   = uniform(-lim, lim)           #bias for hidden node 1
b2   = uniform(-lim, lim)           #bias for hidden node 2'''

h1_sum,h2_sum,out_sum= 0, 0, 0 #outputs before squashing
rate = 0.1

def sig(x):                 #sigmoid activation function
    return 1/(1+exp(-x))

def fwd(array_row):
    global w1, w2, w11, w12, w21, w22
    global h1_sum, h2_sum, out_sum
    global bout, b1, b2
    x1 = array_row[0]
    x2 = array_row[1]
    h1_sum  = w11*x1 + w21*x2 + b1
    h2_sum  = w12*x1 + w22*x2 + b2
    out_sum = w1*sig(h1_sum)+w2*sig(h2_sum)+bout


def bak(array_row):
    global w1, w2, w11, w12, w21, w22
    global h1_sum, h2_sum, out_sum
    global bout, b1, b2
    x1 = array_row[0]
    x2 = array_row[1]
    target  = array_row[2]
    out = out_sum                   #linear output
    #out = sig(out_sum)             ####squashed output
    D = (target-out)                #Delta for linear output
    #D = (target-out)*out*(1-out)   #####Delta for squashed output
    h1, h2 = sig(h1_sum), sig(h2_sum)
    Dw1  = D*h1
    Dw2  = D*h2
    Dw11 = D*w1*h1*(1-h1)*x1
    Dw12 = D*w2*h2*(1-h2)*x1
    Dw21 = D*w1*h1*(1-h1)*x2
    Dw22 = D*w2*h2*(1-h2)*x2
    Dbout = D
    Db1   = D*w1*h1*(1-h1)
    Db2   = D*w2*h2*(1-h2)
    #Do not combine the steps below with the ones above!!
    #We need the old weights w1 w2 w3 above!
    w1   += rate*Dw1
    w2   += rate*Dw2
    w11  += rate*Dw11
    w12  += rate*Dw12
    w21  += rate*Dw21
    w22  += rate*Dw22
    #Uncomment the lines below if you want the biases to adapt, too
    '''bout += rate*Dbout
    b1   += rate*Db1
    b2   += rate*Db2'''
    #end backprop


def predict(arr, verbose=0):
    global w1, w2, w11, w12, w21, w22
    global bout, b1, b2
    if verbose:
        print 'target   binary output   output'
    Error = 0
    same = True
    for i in range(arr.shape[0]):
        x1 = arr[i, 0]
        x2 = arr[i, 1]
        h1 = sig(w11*x1 + w21*x2 + b1)
        h2 = sig(w12*x1 + w22*x2 + b2)
        out = w1*h1 + w2*h2 + bout
        #out = sig(w1*h1 + w2*h2 + w3*h3 + bout) #squashed output
        Error += (arr[i,2]-out)**2
        #The following threshold only works for linear and sigmoid outputs
        #For tanh, it should be compared to 0 instead
        if out < 0.5:              ####squashed output
                bin_out = 0
        else:
                bin_out = 1
        if verbose:
            print('  {:3d} {:10d} \t {:7.5f}'.format(arr[i, 2], bin_out, out))
        same = (same and (arr[i,2]==bin_out))      
    print('  Total Error = {:.8f}'.format(Error))
    return same

    
for i in range(1000000):
    row = i%4    #choose data point deterministically, in a cycle
    #row = randint(0, 3)    #random choice of data point
    print i,
    fwd(andy[row,:])
    bak(andy[row,:])        
    if predict(andy):
        print '############### Success on iteration', i, ' ###########'
        print('outputs:{:.4} {:.4} {:.4}'.format(
            h1_sum, h2_sum, out_sum))
        print('Weights: {:.3} {:.3} {:.3} {:.3} {:.3} {:.3}'.format(
            w1, w2, w11, w12, w21, w22))
        print('Biases:  bout   b1     b2')
        print('        {:5.3f}  {:5.3f}  {:5.3f}'.format(
                bout, b1, b2))
        break
predict(andy, verbose=1)           