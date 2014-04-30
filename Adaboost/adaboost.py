import numpy as np
from numpy import *

def loadSimpleData():
	datMat = matrix([[1.,2.1], 
		[2.,1.1],
		[1.3,1.],
		[1.,1.],
		[2.,1.]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]
	return datMat,classLabels

''' Algo:
	Set the minError to infinite
	For every feature in the dataset:
		for every step:
			for each ineqality:
				build a decision stamp and test it with the weighted dataset
				if the error is less than minError:
						set this stump as the best stump
	Return best stump
'''


################ Decision stump generating funcitons ######################################

''' This function performs a threshold comparison to classify data. Everthing one one side of the threshold is thrown into class -1
    and everything on other side is thrown into class +1 
'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	
	if threshIneq == 'lt' :
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal ] =  1.0
	return retArray	


''' Iteratre over all of the possible inputs to strumpClassify and find the best decision stump for our dataSet '''

def buildStump(dataArr,classLabels,D):
	dataMatrix = mat(dataArr)
	labelMat   = mat(classLabels).T
	m,n	   = shape(dataMatrix)
	numSteps   = 10.0
	bestStump  = {}
	bestClassEst = mat(zeros((m,1)))
	minError   = inf
	
	for i in range(n):
		rangeMin = dataMatrix[:,i].min() 
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax - rangeMin) / numSteps
		
		for j in range(-1,int(numSteps) + 1):
			for inequal in ['lt','gt']:
				threshVal = ( rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
				errArr = mat(ones((m,1)))
				
				errArr[predictedVals == labelMat] = 0
				weightedError  = D.T * errArr
				print "split: dim %d , thresh %.2f, thresh ineqal: %s , the weighted error is %.3f" % (i, threshVal, inequal, weightedError)				
				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClassEst
				


####################################### Full AdaBoost Algorithm #################################################


''' 
For each iteration:
		Find the best stump using buildStump()
		
		Add the best stump to the stump array

		Calculate alpha

		Calculate the new weight error = D
		
		Update the aggregate class estimate

		if the error rate == 0.0:
				break out of the for loop

'''

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr = list()
	m = shape(dataArr) [0]
	D = mat(ones((m,1))/m)
	aggClassEst = mat(zeros((m,1)))
	
	for i in range(numIt):
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)
		print "D:",D.T
		
		alpha = float(0.5*log((1.0 - error) / max(error,1e-16)))
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		

		print "classEst : ", classEst.T
		expon = multiply(-1 * alpha *mat(classLabels).T,classEst)

		D = multiply(D,exp(expon))
		D = D /D.sum()
		
		aggClassEst += alpha * classEst

		print "aggClassEst: ", aggClassEst.T

		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T , ones((m,1)))

		errorRate = aggErrors.sum() / m
		print "total errors: ", errorRate,"\n"

		if errorRate == 0.0:
			break
	return weakClassArr	
	

######################################## Testing /        Classifying with Adaboost #################################

def adaClassify(datToClass, classifierArr):
	dataMatrix = mat(datToClass)
	m = shape((datToClass))[0]
	aggClassEst = mat(zeros((m,1)))
	
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
		print aggClassEst
	return sign(aggClassEst)
		








if __name__=='__main__':
	datamat,classLabels = loadSimpData()

