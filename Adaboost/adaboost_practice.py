import numpy as np
from numpy import *

from adaboost import *



''' This program runs Adaboost on a difficult dataSet 


Steps to follow:

1. Collect : Text file provided 

2. Prepare: We need to make sure the class labels are +1 or -1.

3. Analyze: Manually inspect the data

4. Train: We will train a series of classifiers on the data using the adaboostTrainDS() function

5. Test: We have two datasets. With no randomization, we can have an apples to apples comparison of the Adaboost results versus the logistic regression results

6. Use: We will look at the error rates in this example


'''


########################### Collect ######################################

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t'))
	dataMat,labelMat = list(),list()
		
	fr = open(fileName)
	for line in fr.readlines():
			lineArr = list()
			curLine = line.strip().split('\t')
			for i in range(numFeat -1):
				lineArr.append(float(curLine[i]))
			dataMat.append(lineArr)
			labelMat.append(float(curLine[-1]))
	return dataMat,labelMat


if __name__=='__main__':
	datArr,labelArr= loadDataSet('horseColicTraining.txt')
	classifierArray = adaBoostTrainDS(datArr,labelArr,10)

	testArr,testLabelArr = loadDataSet('horseColicTest.txt')
	prediction10 = adaClassify(testArr,classifierArray)

	errArr = mat(ones((67,1)))
	errArr [ prediction10 != mat(testLabelArr).T].sum()
	
	errorRate = errArr / 67
	print errorRate
