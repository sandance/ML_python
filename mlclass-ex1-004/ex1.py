import numpy as np
from numpy import *
from pylab import scatter,show,title,xlabel,ylabel,plot,contour,axes
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from numpy.linalg import *

############################### Plotting the Data #######################
def plotData(X,y):
	scatter(X,y,marker='o', c='b')
	title('Profits distribution')
	xlabel('Population of city in 10,000s')
	ylabel('Profit in $10,000')
	show()

############################### Compute Cost ############################

''' This will compute the cost of linear regression using (theta) 
    as the parameter for linear regression to fit the data points in X 
    and y '''
def computeCost(X,y,theta):
	m = shape(X)[0]  # number of training examples
	J=0 
	# computation
	Prediction = X * theta
	Errors  = (Prediction - y)
	sqrErrors = power(Errors,2)
	J = sum((sqrErrors)) /(2 * m)
	return J

################################ Gradiant Descent ########################

''' In this section, we will fit the linear regression parameter (theta) to our dataset
    using gradiant descent . The main objective of linear regression is to minimize the cost
    Function. i.e minimizing value of J(theta) by changing the value of (theta) , not by X and y '''

def gradientDescent(X,y,theta,alpha,iterations):
	# Taking iterations gradient steps with learning rate alpha 
	# Initialize some useful values
	m = shape(X)[0]	
	J_history = zeros((m,1))
	for i in range(m):
	# While debugging it can be useful to print out the values   of the cost function (computeCost and Gradiant here)
		#temp1 = theta[0] - (alpha * (1/m) * sum((h-y)))
		#temp2 = theta[0] - ( (alpha * (1/m) ) * (sum((h-y))* X(:,1))) 		
		predictions= X.dot(theta).flatten()	# equivalent to h=X*theta
		error_x1 = (predictions - y).T * X[:,0]
		error_x2 = (predictions - y).T * X[:,1]
		
		theta[0][0] = theta[0][0] - alpha * (1.0 /m) * error_x1.sum()
		theta[1][0] = theta[1][0] - alpha * (1.0 /m) * error_x2.sum()
		print "Theta values : %f %f\n" % (theta[0][0], theta[1][0])		
		J_history[i,0] = computeCost(X,y,theta)
	return theta,J_history
		




################################  Main Function ##########################

if __name__ == '__main__':
	# load the data using numpy 
	# First column refers to pupulation in $10,000
	# Second column refers to profit in $10,000
	data = np.loadtxt('ex1data1.txt', delimiter=',')
	X = mat(data[:,0]).T
	y = mat(data[:,1]).T
	m=shape(X)[0] 			# number of training examples
	
	###################### Plotting data #########################
	print "Plotting data\n"
	plotData(X,y)
	raw_input('Press any key to exit\n')
	[m,n]=shape(X)
	#X = c_[np.ones(m),X] 	# Add a one matrix in X
	new_col=ones((m,1))
	X = np.hstack((new_col,X))
	theta = np.zeros((2,1))
	
	# some gradient descent settings
	iterations = 1500
	alpha = 0.0001 # learning rate
	# initial_cost
	initial_cost = computeCost(X,y,theta)
	print ' initial cost is %f ' % initial_cost	
	# run Gradiant descent 
	theta,J_hist = gradientDescent(X,y,theta,alpha,iterations)
	print theta
	# print theta to screen 
	print 'Theta found by gradient descent: \n'
	print '%f %f \n' % (theta[0],theta[1])		
	##########################                 ######################
	# Predict values for population sizes for 35000 and 70000 
	predict1 = array([1,3.5]).dot(theta).flatten()
	print "For population =35000 we predict a profit of %f" % ( predict1 * 10000)
	predict2 = array([1,7]).dot(theta).flatten()
	print "For population = 70000 we predict a profit of %f" % ( predict2 * 10000)

	####################### Plot the results   ################################
	print "Visualizing J(theta_0,theta_1) ...\n"
	
	# Grid over which will calculate J
	theta0_vals = linspace(-10,10,100)
	theta1_vals = linspace(-1,4,100)
	
	# Initialize J_vals to a matrix of 0's
	J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))
	
	# Fill out J_vals
	for i, element1 in enumerate(theta0_vals):
		for j,element2 in enumerate(theta1_vals):
			thetaT = zeros(shape=(2,1))
			thetaT[0][0],thetaT[1][0] = element1, element2
			J_vals[i,j] = computeCost(X,y,thetaT)

	#contour plot
	# Because of the way meshgrids work in the surf command, we need to 
	# transpose J_vals before calling surf, or else the axes will be flipped	
	J_vals = J_vals.T 
	
	# Plot J_vals as 15 contour spaced logarithmacally 
	contour(theta0_vals,theta1_vals, J_vals , logspace(-2,3,20))
	xlabel('theta_0')
	ylabel('theta_1')
	scatter(theta[0][0],theta[1][0])
	show()	
	raw_input("press any key to exit ")
