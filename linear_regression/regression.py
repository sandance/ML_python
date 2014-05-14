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
			
