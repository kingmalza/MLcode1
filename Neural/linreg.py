
#K!ngMalza '17

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

import numpy as np

def linreg(inputs,targets):

	inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
	beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)

	outputs = np.dot(inputs,beta)
	#print shape(beta)
	#print outputs
	return beta
