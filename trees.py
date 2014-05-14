from numpy import *
import numpy as np

"""
	Pseudo-code for createBranch()
	1.Check if every item in the dataset is in the same class
		2. If so return the class level
		
		else
			find the best feature to split the data
			split the dataset
			create a branch node
				for each split 
					call createBranch and add the result
		return branch node

"""

from math import log

# Calculate count of number of instances in the dataset 

def calcShanonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys(): #if not in dictionary, add one 
			labelCounts[currentLabel] =0
		labelCounts[currentLabel] +=1
	shanonEnt=0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) /numEntries
		shanonEnt -= prob * log(prob,2)
	return shanonEnt



def createDataSet():
	dataSet =  [[1,1,'yes'],
		    [1,1,'yes'],
		    [1,0,'no'],
		    [0,1,'no'],
		    [0,1,'no']]
	labels =['no surfacing','flippers']
	return dataSet, labels




"""   DataSet splitting on a given feature

      dataSet = Data Set which one we will split
      axis    = The feature we'll split on
      value   = Value of the feature to return 

"""

def splitDataSet(dataSet,axis,value):
	retDataSet =[]
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
	#		print "ReducedFeatVec " 
	#		print reducedFeatVec
			reducedFeatVec.extend(featVec[axis+1:])
	#		print "ReducedFeatVec after extend" 
	#		print reducedFeatVec
			retDataSet.append(reducedFeatVec)
	#		print "RetDataSet "  
	#		print retDataSet
	return retDataSet

# Choosing the best features to split on

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) -1
	baseEntropy = calcShanonEnt(dataSet)
	bestInfoGain = 0.0 ; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		print "for each i=%d featlist is = \n" % i,featList
		uniqueVals = set(featList)
		newEntropy= 0.0
		for value in uniqueVals:
			""" Calculate entrophy for each split, here  i is considered each feature in the dataset and value is each row in the dataset """
			subDataSet = splitDataSet(dataSet,i,value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShanonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):   # Finding best information gain 
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature



import operator

def majorityCnt(classList):
	classcount={} # dictionary 
	for vote in classList:
		if vote not in classcount.keys():
			classcount[vote]=0
		classCount[vote] +=1
	sortedClassCount = sorted(classCount.iteriterms(),key=operator.itemgetter(1),reverse=True)
	print "SortedClasscount %d \n" , sortedClassCount
	return sortedClasscount [0][0]

	

#Building Trees 

def createTree(dataset,labels):
	classList = [example[-1] for example in dataset ]
	print classList
	if classList.count(classList[0]) == len(classList): # Stop when all the classes are equal
			return classList[0]
	if len(dataset[0]) == 1: # When no more features, return majority 
			return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataset)
	print bestFeat
	bestFeatLabel=labels[bestFeat]
	myTree={ bestFeatLabel:{} }
	del(labels[bestFeat])
	#finally you iterate over all the unique values and recursively call createTree() for each split  of the dataset
	featValues=[example[bestFeat] for example in dataset ]
	print "FeatValues \n"
	print featValues
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels=labels[:] # makes a copy of labels and places it in a new list called subLabels
		myTree[bestFeatLabel][value] = createTree(splitDataSet \
					 (dataset,bestFeat,value) , subLabels)
	return myTree	
