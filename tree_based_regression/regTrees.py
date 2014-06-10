from numpy import *

''' General approach to tree based regression 
	1. Collect : Any method
	2. Prepare : Numeric values are needed. If you have norminal values, its good idea to map them into binary values
	3. Analyze : We will visualize the data into 2D plot and generate trees as dictionaries
	4. Train   : The majority of the time will be spent building trees with models at the leaf nodes
	5. Test	   : We will use R^2 value with test data to determine the quality of our models
	6. Use 	   : We will use our trees to make forcasts. 

'''

############################### Building Trees with continuous and discrete features #################

class treeNode ():
	def __init__ (self, feat,val,right,left):
		featureToSplitOn = feat 
		valueOfSplit     = val
		rightBranch	 = right
		leftBranch	 = left



################################ Create Tree ######################################

# pseudo code
'''
	find the best feature to split on:
		If we cant split the data, this node becomes leaf node
		Make a binary split of the data
		
		Call createTree() on the right split of the Data
		Call createTree() on the left split of the Data
'''

def loadDataSet(fileName):
	dataMat = list()
	fr  = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine) 			# Map everything to float
		dataMat.append(fltLine)
	return dataMat


''' This function takes three arguments : a dataSet, a feature on which to split and a value for that feature 
    and the function returns 2 sets. The two sets are created using array filtering for the given feature and value '''


def binSplitDataSet(dataSet, feature,value):
	mat0 = dataSet[nonzero(dataSet[:,feature] >  value)[0],:][0]
	mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
	return mat0,mat1



''' 
createTree() is a recursive function that first attempts to split the dataset into two parts



''' 

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]



def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	feat, val = chooseBestSplit(dataSet,leafType,errType,ops)
	if feat == None:
		return val
	retTree = dict();
	retTree['spInd'] = feat
	retTree['spVal'] = val
	
	lSet,rSet = binSplitDataSet(dataSet,feat,val)
	retTree['left'] = createTree(lSet,leafType,errType,ops)
	retTree['right'] = createTree(rSet,leafType,errType,ops)
	return retTree


####################### chooseBestSplit() ########################

''' Algo: 
	For every feature:
		For every unique value:
			Split the dataSet into two
			Measure the error of the two splits
				
			If the error is less than BestError:
				Set bestSplit to this split and update bestError
	Return bestSplit feature and treashold
'''

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	tolS = ops[0] ; tolN = ops[1]
	# Exit if all values are equal

	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)
	
	m,n = shape(dataSet)
	  S = errType(dataSet)
	  
	bestS = inf ; bestIndex = 0; bestValue = 0
	for featIndex in range(n-1):
		for splitVal in set(dataSet[:,featIndex]):
			mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
				
	













