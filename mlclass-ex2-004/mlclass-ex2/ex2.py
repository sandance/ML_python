import numpy as np
from numpy import *
import operator
import scipy
from scipy import optimize
from numpy import newaxis, r_, c_, mat, e
from numpy.linalg import *
import matplotlib.pyplot as plt

################################# Loading Data ##############################

#Instead of doing all , i just could use np.loadtxt('ex2data1.txt',delimiter=',')

def load_data(filename):
	# open the file
	data = open(filename)
	n = len(data.readlines())
	print n
	X = zeros((n,2)) ; y  = zeros((n,1))
	index=0
	data = open(filename)
	for line in data.readlines():
		line = line.strip()
		listFromline = line.split(',')
	#	print listFromline
		X[index,:] =listFromline[0:2]
		y[index,:] =listFromline[2]
		index +=1	
	return X,y



################################ Plotting data #############################

''' Alternate plot createor from  X,y points only '''

def plotFromData(X,y):
	#import matplotlib.pyplot as plt
	pos = (y == 1).nonzero()[:1]
	neg = (y == 0).nonzero()[:1]
	plt.plot(X[pos,0].T,X[pos,1].T,'k+',markeredgewidth=2,markersize=7)
	plt.plot(X[neg,0].T,X[neg,1].T,'ko',markerfacecolor='r',markersize=7)
	



''' This will create plot from data itself '''
def plotData(filename):
	#import matplotlib.pyplot as plt 
	print "Plotting data with + indicarting (y=1) examples and \
		o indicating (y=0) examples \n"
	X,y=load_data(filename)
	dataArr = array(X)
	n_total = shape(dataArr)[0]
	xcord1,ycord1 = list(),list()
	xcord2,ycord2 = list(),list()
	for i in range(n_total):
		if int(y[i]) == 1:
			xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
		else:
			xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x = arange(-3.0,30,0.1)
	y = arange(-3.0,30,0.1)
	#ax.plot(x,y)
	plt.xlabel('Exam 1 Score'); plt.ylabel('Exam 2 Score')
	plt.legend(['Admitted','Not admitted'])
	plt.show()
################################ Sigmoid Function ################################

def sigmoid(inX):
        return 1.0/(1+exp(-inX))
		

################################ Compute Cost and Gradient ########################
'''
In this part , we will implement the cost and gradient for logictic regression . need 
to write a function naed costFunction.m 
'''

def compute_grad(X,y,theta):
	m=len(y) # size of training examples
	grad=zeros(shape(theta)) # initialize gradiant descent
	h=X.dot(theta)
	grad = transpose((1./m)*transpose(sigmoid(X.dot(theta)) - y).dot(X))
	return grad


def costFunction(X,y,theta):
	# This function takes X, y and fitting parameter theta values 
	# and it returns J , cost of using theta as the parameter for logistic regression
	# grad , as gradient of the cost w.r.t parameters
	m=len(y) # size of training examples 
	J=0
	grad=zeros(shape(theta)) # initialize gradiant descent
	#J = 1 ./m * (-y' * log(sigmoid(X*theta)) - (1 - y' ) * log(1 - sigmoid(X*theta)))
	# grad = 1 ./m *( X' * (sigmoid(X*theta) - y))
	h=dot(X,theta)
	# calculate cost
	#J = (dot(-y.H,np.log10(sigmoid(h))) - dot(( 1 - y.conj().T),np.log10(1 - sigmoid(h))))
	J = (1./m) * (-transpose(y).dot(log(sigmoid(X.dot(theta)))) - transpose(1-y).dot(log(1-sigmoid(X.dot(theta)))))
	# calculate gradiant descent
	#grad = 1/m * dot( X.conj().T,(sigmoid(h) - y))
	grad = transpose((1./m)*transpose(sigmoid(X.dot(theta)) - y).dot(X))
	return J[0]##,grad , dont return grad as fminc only accpets one 

#################################### Optimizing using fminunc #######################
''' In this exercise, we will use built in function (fminunc) to find the optimal parameters theta '''
#	options = {'full_output': True, 'maxiter': 400 }
	 


############################### Plottting Decision Boundary ###########################

''' This will plot the decision boundary in the graph using the cost and theta value obtained from fminuc '''

def plotDecisionBoundary(X,y,theta):
	plotFromData(X[:,1:3],y)
	
	if X.shape[1] <= 3: #if colums in X is 3 or less	
		# Only need 2 points to define a line , so choose two endpoints
		plot_x = r_[X[:,2].min()-2, X[:,2].max()+2]
		# Calculate the decision boundary line 
		# actual octave implemetation is : plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
		plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
		
		# plot  and adjust axes for better viewing 
		plt.plot(plot_x,plot_y)
		plt.legend(['Admitted','Not admitted','Decision Boundary'])
		plt.axis([20,100,20,100])
	else:
		pass

################################ Evaluating Logictic Regression #####################
''' After learning the parameters, we need to use the model to predict whether particular
    student will be admitted . This script will procced to report training accuracy of classifier
    by computing percentage of examples it got correct '''
def predict(theta,X):
	m=shape(X)[0] # number of training examples
	h=X  * theta.T
	p =sigmoid(h) >= 0.5 
	return p

############################### Main Function ########################################
if __name__ == "__main__":
	X,y=load_data('ex2data1.txt')
	data = open('ex2data1.txt')
        m = len(data.readlines())
#	plotData('ex2data1.txt')
	[a,b] = shape(X)
	# Add a Column  to  X
	new_col=ones((a,1)) # this will create one column with all 1's
	# modified X is
	X_modi = np.hstack((new_col,X))
	#Initialize fitting parameters , i mean the Theta
	initial_theta=zeros((shape(X_modi)[1],1)) # initialize all theta values	
	J,grad = costFunction(X_modi,y,initial_theta),None
	print 'Cost at initial theta (zeros): %f' % J
	print 'Gradiant at initial theta (zeros):%s' % grad
	# Now , here we will build optimization function to  find optimal parameters of theta
	options = {'full_output': True , 'maxiter' : 400 }
	theta,J,_,_, _ = optimize.fmin(lambda t: costFunction(X_modi,y,t), initial_theta,**options)
	print 'Cost at theta found by fminunc: %f' % J
	print 'theta: %s' % theta
	raw_input('Press any key to continue\n')
	############ After training the data plot the decision boundary ###############
	plotDecisionBoundary(X_modi,y,theta)
	plt.show()
	raw_input('Press any key to continue\n')
	plt.close(1)
	##################################### Prediction and Accuracy ##################
	
	# After learning the parameters, we'll like to use it to predict the outcomes on unseen data.
	# In this part, we will use the logistic regression model to predict the probability that a student
	# with score 45 on exam1 and score 85 on exam2 will be admitted 
	
	# We will also compute the training and test set accuracies of our model
	# Predict probability for a student with score 45 on exam1 and score 85 on exam 2
	prob = sigmoid(mat('1 45 85') * c_[theta])
	print 'For a student with scores 45 and 85, we predict an admission probability of %f' % prob
	p = predict(theta,X_modi)
	print 'Train accuracy:' 
	answer= (p == y).mean() * 100
	print answer
#	print 'Train accuracy: %f' % ((y[np.where(p==y)].size / float(y.size)) * 100)
	raw_input('Press any key to continue\n') 	
	
