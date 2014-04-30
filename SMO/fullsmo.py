import numpy as np
from numpy import *


# Data Structure for Full patt SMO

class optStruct:
	def __init__(self,dataMatIn,classLabels,C,toler):
		self.X = X
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn) [0]
		self.alphas = mat(zeros((self.m,1)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2)))
		

''' Calculates error value '''
def calcEk(oS,k):
	fXk = float(multiply(oS.alphas,oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
	Ek  = fXk - float(oS.labelMat[k])
	return Ek



''' SelectJ selects the second alpha, or the inner loop alpha, The goal is to choose the second alpha so that  we will
take maximum step during each optimization. This function takes the error value associated with the first choice alpha (Ei) and the index i '''	

def selectJ(i,oS,Ei):
	maxK = -1
	maxDeltaE =0
	Ej = 0
	oS.eCache[i] = [1,Ei]
	validEcacheList = nonzero(os.eCache[:,0].A)[0]
	if (len(validEcacheList)) > 1:
		for k in validEcacheList:
			if k == i:
				continue
			Ek = calcEk(oS,k)
			deltaE = abs(Ei -Ek)
			if (deltaE > maxDeltaE):
				maxK = k
				maxDeltaE = deltaE
				Ej = Ek
		return maxK,Ej



