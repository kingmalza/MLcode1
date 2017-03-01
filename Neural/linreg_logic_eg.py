
#K!ngMalza '17

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Demonstration of the Perceptron and Linear Regressor on the basic logic functions

import numpy as np
import linreg

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
testin = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)

# AND data
ANDtargets = np.array([[0],[0],[0],[1]])
# OR data
ORtargets = np.array([[0],[1],[1],[1]])
# XOR data
XORtargets = np.array([[0],[1],[1],[0]])

print "AND data"
ANDbeta = linreg.linreg(inputs,ANDtargets)
ANDout = np.dot(testin,ANDbeta)
print ANDout

print "OR data"
ORbeta = linreg.linreg(inputs,ORtargets)
ORout = np.dot(testin,ORbeta)
print ORout

print "XOR data"
XORbeta = linreg.linreg(inputs,XORtargets)
XORout = np.dot(testin,XORbeta)
print XORout
