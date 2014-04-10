import operator
from operator import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy 
from scipy import optimize
from numpy.linalg import *
from pylab import scatter,show,title,xlabel,ylabel,plot,contour,axes


##################################  Feature Normalization #########################
'''
	featureNormalize(X) returns a normalized version of X where the mean value 
	of each feature is 0 and standard deviation is 1 . This is often good preprocessing
	step while working with learning Algorithms
'''

def featureNormalize(X):
	X_norm = X.copy()
	m = shape(X)[0]
	mu = zeros((1, shape(X)[1]))
	sigma = zeros((1, shape(X)[1]))
	sigma = X.std(axis=0)	
	indicies = shape(X)[1]
	mu = X.mean(axis=0)
	result = (X_norm - mu ) / sigma
	X_norm = result.copy()
	return X_norm, mu, sigma



################################ Cost Function with multi variable ##################

'''
	Computes the cost of using theta as the parameter for linear regression
	to fit the data points in X and Y 
'''

def computeCost(X,y,theta):
	m = shape(y)[0]
	Prediction = X * theta
	sqrError = (Prediction - y) 
	
	J = ( 1.0 / ( 2*m)) * sqrError.T.dot(sqrError)
	return J

###################################### Gradiant Descent with multi variable #########

'''
Gradiant Descent is being used to learn theta
this function updates theta by taking num_iters gradient steps with learning rate alpha
'''

def gradiantDescentMulti(X,y,theta,alpha,num_iters):
	m = shape(y)[0]	 # num of training examples
	J_history = np.zeros((num_iters,1))
	[a,b] = shape(theta)
	
	for i in range(num_iters):
		prediction = X.dot(theta)
		cost = (prediction - y)
		for j in range(a):
			temp = X[:,j]
			errors_x1 = cost * temp.T
			theta[j][0] = theta[j][0] - alpha * (1.0/ m) * errors_x1.sum()
		
		J_history[i,0] = computeCost(X,y,theta)
	return theta,J_history					






###################################### Plotting the Data #############################

#def plotData(X,y):
	



###################################### Main Function ###################################

if __name__ == '__main__':
	# Load the data using numpy
	data = np.loadtxt('ex1data2.txt', delimiter=',')
	X = mat(data[:,:2])
	y = mat(data[:, 2]).T
	m = shape(X)[0] 		# number of training examples
	# Scale features and set them to zero mean
	X,mu,sigma = featureNormalize(X)
	new_col=ones((m,1))
	# Add a column to X
	X = np.hstack((new_col,X))
	############################### Running Gradient Descent #######################
	print " Running Gradient Descent .......\n"
	# choose some alpha value
	best_value = dict()
#	alphas = [0.01,0.001,0.0001,0.1,0.005]
	alpha=0.01
	num_iters = 400
	# Init Theta and Run Gradiant Descent
	theta = np.zeros((3,1))
	#for alpha in alphas:
	theta,J_history = gradiantDescentMulti(X,y,theta,alpha,num_iters)
	print theta
	#best_value[alpha] = theta
	############################## Plot the Data ###################################

	plot(arange(num_iters),J_history)
	xlabel('Iterations')
	ylabel('cost Function')
	show()
	raw_input("Press any key to exit.....")

	##############################  Estimate the price of a 1500 sq ft, 3 br house ###
	# Predict for 1500 sq ft and 3 br hourse
	predict_price = array([1.0,((1600 - mu[0]) / sigma[0]),((3 - mu[1]) / sigma[1])])*theta
	print "For 1650 sq feet , 3 bed room ,  we predict house price %s" % (predict_price)
		
