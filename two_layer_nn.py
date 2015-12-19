import numpy as np 

def nonlin(x, deriv=False):
	if deriv:
		return x * (1 - x)
	return 1/(1 + np.exp(-x))


# The training features
X = np.array([ [0, 1, 0],
				[1, 1, 0],
				[1, 0, 1],
				[0, 1, 1],
				[1, 0, 0] ])

# The labels
y = np.array([[0, 1, 1, 0, 1]]).T

bias = np.ones((X.shape[0], 1))
X_feature = np.concatenate((bias, X), axis=1)

np.random.seed(0)
syn0 = np.random.random((X_feature.shape[1], 1))

for i in xrange(10000):
	l0 = X_feature
	l1 = nonlin(np.dot(l0, syn0))

	l1_error = y - l1 

	l1_delta = l1_error * nonlin(l1, True)

	syn0 += np.dot(l0.T, l1_delta)

print "The prediction after iteration is"
print l1 
