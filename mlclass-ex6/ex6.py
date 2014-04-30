import numpy as np
from numpy import *
import scipy,scipy.io
from scipy.io import loadmat
from scipy import *
import matplotlib.pyplot as plt

##################################### Part 1: Loading and Visualizing Data ####################


''' Load the Matrix data into Python array '''
def loadData():
	data = loadmat('ex6data1.mat');
	X=data['X']
	y=data['y']
	return X,y



''' Plot the X,y co-ordinates into matplotlib '''
def plotData(X,y):
	pos = (y == 1).nonzero()[:1]
	neg = (y == 0).nonzero()[:1]
	plt.plot(X[pos,0].T,X[pos,1].T,'k+',markeredgewidth=2,markersize=7)
	plt.plot(X[neg,0].T,X[neg,1].T,'ko',markerfacecolor='r',markersize=7)


#################################### Part 2: Training Linear SVM ############################

'''The following code will train a linear SVM on the dataset and plot the decision boundary learned  '''




if __name__ == "__main__":
	X,y=loadData()
	#print X
	plotData(X,y)
	plt.show()
	raw_input('Press any key to continue\n')
	plt.close(1)
