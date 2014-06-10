from numpy import *

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat = list() ; labelMat = list()
	
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat


def standRegres(xArr,yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	xTx = xMat.T * xMat
	
	if linalg.det(xTx) == 0.0:
		print "This matrix is singular, can not do inverse"
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws
			
def plot(xMat,yMat):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0] , yMat.T[:,0].flatten().A[0])	
	
############################################## Locally weighted linear regression #######################################

''' linear regression has the tends to underfit the data. one way to get rid of it , is using locally weighted linear regression (LWLR).
    in LWLR we give a weight to data points near our data point of interest , then we compute a least squares regression '''

def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat = mat(xArr); yMat = mat(yArr).T
	m = shape(xMat)[0]
	weights = mat(eye((m)))
	for j in range(m):
		diffMat = testPoint - xMat[j,:]
		weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx = xMat.T * (weights * xMat)
	if linalg.det(xTx) == 0.0:
		print "This matrix is singular , can not do inverse"
		return
	ws = xTx.I * (xMat.T * (weights * yMat))
	return testPoint * ws



def lwlrTest(testArr, xArr, yArr, k=1.0):
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat


def rssError(yArr,yHatArr):
	return ((yArr - yHatArr) ** 2).sum()


''' Regression 

if __name__ =='__main__':
	xArr,yArr = loadDataSet('ex0.txt')
	ws = standRegres(xArr,yArr)
	
	xMat = mat(xArr)
	yMat = mat(yArr)
	yHat = xMat*ws
	plot(xMat,yMat)
	
'''

# For Abanole testing

if __name__=='__main__':
	abx,aby = loadDataSet('abalone.txt')
	yHat = lwlrTest(abx[0:99],abx[0:99],aby[0:99],0.1)



################################################### Ridge Regression ##################################################

''' If you have more features than actual data points, then making a prediction using linear regression may not work, for that we have a method called Ridge Regression , in ridge regression we put a additional matrix (lambda)I to the matrix xTx . '''

def ridgeRegres(xMat,yMat,lam=0.2): # Default value is 0.2 , if no other value is passed to the function 
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam 
    
    if linalg.det(denom) == 0.0 :
            print "This matrix is singular, cannot do inverse"
            return 
    ws = denom.I * (xMat.T * yMat)
    return ws

''' Here we are normalizing our data to give each feature equal importance regardless of the units it was measured in . This is
    done by subtracting off the mean from each feature and dividing by the variance . 

    After the regularization is done, you call ridgeRegres() with 30 different lambda values. the values vary exponentially so
    now we can see , how very small values of lambda and very large values impact your results. '''

    
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMean = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMean) / xVar
    numTestPts = 30 
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat


   
############################### StageWise linear Regression #####################################

''' It can be shown that the equation for ridge regression is the same as out regular least squares regression using lasso
    or Forward Stagewise Regression 
	
    Forward Stagewise Regression:
		This algorithm is a greedy algorithm in that at each step it makes the decision that will reduce the error the 
		most at that step.

		Algo:
			Regularize the data to have 0 mean and unit variance 
			
			For every iteration:
				Set lowestError to  INFINITE
				For every feature:
					For increasing and decreasing:
						change one coefficient to get a new W
						Calculate the error with new W
					
						If the error is lower than lowestError:
							set Wbest to the current W
				Update set W to Wbest
'''


def regularize(xMat):#regularize by columns
	inMat = xMat.copy()
	inMeans = mean(inMat,0)
	inVar = var(inMat,0)      #calc variance of Xi then divide by it
	inMat = (inMat - inMeans)/inVar
	return inMat







def stageWise(xArr,yArr,eps=0.01,numIt=100):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	### Regularization
	yMean = mean(yMat,0)
	yMat  = yMat - yMean
	xMat  = regularize(xMat)
	m,n   = shape(xMat)
	
	ws    = zeros((n,1))
	wsTest = ws.copy()
	wsMax = ws.copy()
	returnMat = zeros((numIt,n))

	for i in range(numIt):
		print ws.T
		lowestError = inf
		for j in range(n):
			for sign in [-1,1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign
				yTest = xMat * wsTest
				rssE  = rssError(yMat.A ,yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:] = ws.T
	return returnMat 
	
	
######################################### Cross Validation ########################################

def crossValidation(xArr,yArr,numval=10):
	m = len(yArr)
	indexList = range(m)
	errorMat = zeros((numVal,30))
	
	for i in range(numVal):
		trainX = [] ; trainY = []
		testX  = [] ; testY  = []
		random.shuffle(indexList)
		
		for j in range(m):
			if j < m*0.9:
				trainX.append(xArr[indexList[j]])
				trainX.append(yArr[indexList[j]])
			else:
				testX.append(xArr[indexList[j]])
				testX.append(yArr[indexList[j]])
		wMat = ridgeTest(trainX,trainY)
	

		# Regularize test with training params
		for k in range(30):
			matTestX   = mat(testX)
			matTrainX  = mat(trainX)
			meanTrain  = mean(matTrainX,0)
			varTrain   = var(matTrainX,0)
			matTestX   = (matTestX - meanTrain) / varTrain
			yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
			errorMat[i,k] = rssError(yEst.T.A, array(testY))
		meanErrors = mean(errorMat,0)
		minMean    = float(min(meanErrors))
		bestWeights = wMat[nonzero(meanErrors==minMean)]
		xMat  = mat(xArr); yMat = mat(yArr)
		meanX = mean(xMat,0); varX = var(xMat,0)
		unReg = bestWeights / varX
		print "The best model from ridge regression is:\n", unReg
		print "with constant term: " , -1 * sum(multiply(meanX,unReg)) + mean?(yMat)		



















 
