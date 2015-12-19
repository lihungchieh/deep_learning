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
				[1, 0, 0],
				[0, 0, 0],
				[1, 1, 1] ])

# The labels
y = np.array([[0, 0, 1, 1, 1, 0, 0]]).T

bias = np.ones((X.shape[0], 1))
X_feature = np.concatenate((bias, X), axis=1)

np.random.seed(0)
syn0 = 2 * np.random.random((X_feature.shape[1], 4)) - 1
syn1 = 2 * np.random.random((syn0.shape[1]+1, 1)) - 1

for i in xrange(60000):
	l0 = X_feature

	l1 = nonlin(np.dot(l0, syn0))

	# Add bias to each layer
	l1_with_bias = np.concatenate((bias, l1), axis=1)

	l2 = nonlin(np.dot(l1_with_bias, syn1))
	
	if i % 5000 == 0:
		print "The error is: " + str(np.mean(np.abs(y - l2))) 
	# calculate the error of l2
	l2_error = y - l2
	l2_delta = l2_error * nonlin(l2, True)

	l1_error = l2_delta.dot(syn1.T)
	# remove the weight of bias factor, since there is no conbutrion from the previous layer
	l1_delta = l1_error[:,1:] * nonlin(l1, True)

	# print "l1_delta shape is :" + str(l1_delta.shape)
	# print "l0 shape is " + str(l0.shape)
	# print "syn0 shape is " + str(syn0.shape)

	# Update the weight for each layer
	syn1 += np.dot(l1_with_bias.T, l2_delta)
	syn0 += np.dot(l0.T, l1_delta)

print "The prediction after iteration is"
print l2
