import numpy as np
from numpy import *


'''
	loading the txt data into a matrix 
'''

def loadDataSet (fileName):
	dataMat,labelMat = list(),list()
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat

''' Takes two values , first one is the index of our first 
	alphas  and m is the total number of alpha'''
def selectJrand(i,m):
	j=i
	while(j==i):
		j = int(random.uniform(0,m))
	
	return j


''' Clips alphas values, that are greater than H or less than L '''

def clipAlpha(aj,H,L):
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj


		
		
''' 
Pseudocode SMO Algorithm:
	Create an Alphas vector filled with 0's
	While the number of iterations is less than Maxiterations:
		For every data vector in the dataSet:

			if the data vector can be optimized:
				Select another data vector at random
				Optimize the two vectors together
				
				If the vectors can not be optimized:
					break
			if no vector were optimized:
				increment the iteration count




Input:
	C 		= Regularization parameter
	toler 		= Numerical Tolerance
	classLabels 	= y value 
	dataMatrixIn	= X matrix value


output:
	a = Lagrange multiplier for solutions
	b = threshold for solution


'''




# Here is how SMO algorithms works, it chooses two alphas to optimized on each cycle
# once a suitable pair of alphas found , one is increased and one is decreased

#Now alphas needs to meet some criteria
	# One criterion is a pair must meet in that both of the alphas have to be outside their margin boundary
	# Alphas's arenot already clamped or bounded


def smoSimple(dataMatrixIn, classLabels, C, toler, maxIter):
	print "C = %f and toler = %f" % (C,toler)
	dataMatrix = mat(dataMatrix)
	labelMat   = mat(classLebels).transpose()
	b =0
	m,n = shape(dataMatrix)
	alphas = mat(zeros((m,1)))
	num_iter = 0
	
	# while number of iterations is less than maximum iterations
	while(num_iter < maxIter):
		alphasPairsChanged = 0 
		for i in range(m):
			# Enter optimization , if alphas can be changed
			# Here, Equation is F(x) = alphas(i) * y(i) * (x(i),x) + b                check equation 2 in page 1 (Simplified SMO)
			# This will calculate optimal value of W , y =1 only if W is bigger than zero
			# Here  b is something like Theta(0) in logistic Regression 
			
			fXi = float(multiply(alphas,labelMat).T * (dataMatrix[i,:].T )) + b 
			# Calculate The Error value 
			Ei  = fXi  - float(labelMat[i])
			
			# y(i) * Ei < -tol && alphas < C
			
			if ((labelMat[i] * Ei < - toler) and (alphas[i] < C)) or \
				((labelMat[i]*Ei > toler ) and (alphas[i] >0 )):
						# Here, we select a(i) or alphas(i) number
						j = selectJrand(i,m)
						print "random value %f " % j
						# Calculate Ej = f(x(j)) - y(i) 
						fXj = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b 
						Ej  = fXj - float(labelMat[j])
						
						# Save old alphas's . alpha(i)old = alpha(i) and alpha(j) old= alpha(j)
						alphasIold = alphas[i].copy()
						alphasJold = alphas[j].copy()
						
						#Guarantee alphas stay between 0 and C
						if (labelMat[i] != labelMat[j]):
							#Compute L and H , these gives gurantee that alphas stays in between 0 and C
							# We want L<= alphas(j) <= H then a(j) will satisfy 0 <= alpha(j) <= C
								L = max(0,alphas[j] - alphas[i])
								H = min(C,C + alphas[j] - alphas[i])
						else:
								L = max(0,alphas[j] + alphas[i] -C)
								H = min(C , alphas[i] + alphas[j])
						
						# L and H are being calculated , which are being used for clamping alpha[j]
						# if L==H, dont need to change anything		
						if L==H:
							print " L == H" ; continue
						
						# Compute (eta) 
						# Equation (14) on the book
						
						# Eta is the optimal amount to change alpha[j] 
			
						eta = 2 * dataMatrix[i,:] * dataMatrix[j,:].T - \
						dataMatrix[i,:] * dataMatrix[i,:].T - \
						dataMatrix[j,:] * dataMatrix[j,:].T 
						
						if eta >= 0:
							print "eta >= 0"  
							continue 
						# Compute and clip new value for alphas(j) using (12) and (15)
						# if alphas value end up lying outside of L and H 
						alphas[j] -= labelMat[j] * (Ei - Ej) / eta
						alphas[j]  = clipAlpha(alphas[j],H,L)
							
						# if |(a(j) - a(j) old| < 10^-5 
						if (abs(alphas[j] - alphaJold) < 0.00001 ):
								print "J not moving enough" 
								continue
						# Compute alphas value using equation (16)
						# alphas(i) = alpha(i) + y(i) * y(j) (a(j)old -a(j))	
						alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
							 
						# Compute b1 and b2 using (17) and (18) respectively
						#After optimizing alphas(i) and alpha(j) , we select the threashold "b" such that KKT conditions
						# are satisfied for ith and jth example. If, after optimization alphas(i) is not at bounds ( 
						# 0 < alphas(i) < C then b1 is valid

						b1 = b - Ei - labelMat[i] * ( alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - \
						     labelMat[j] * ( alphas(j) - alphaJold ) * dataMatrix[i,:] * dataMatrix[j,:].T

						b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas(j) - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
					
						if   (0 < alphas[i]) and (C > alphas[i]):
							b = b1
						elif (0 < alphas[j]) and ( C > alphas[j]):
							b = b2
						else:
							b = (b1 + b2) / 2.0
						alphasPairsChanged +=1
					
						print "iter: %d i:%d, pairs changed %d" % (iter,i,alphasPairsChanged)
		if (alphasPairsChanged == 0):
			iter +=1
		else:
			iter = 0
		print "Iteration number : %d" % iter
	return b,alphas	
								

##################################################### Patts SMO algorithm ###########################################################


class optStruct:
	def __init__(self,dataMatIn,classLabels, C, toler):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2)))	# Error Cache
		self.K = mat(zeros((self.m,self.m)))
		#for i in range(self.m):
		#	self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)	




''' calcEk() , calculates an E value for a given alphas and returns the E value '''


def calcEk(os,k):
	fXi = float(multiply(os.alphas, os.labelMat).T * (os.X * os.X[k,:].T)) + os.b
	Ek  = fXi - float(os.labelMat[k])
	return Ek 


''' This function selects the 2nd alphas , or the inner loop alpha . the goal is to choose second alpha so that we will take the 
    maximum step during each optimization . This function take the error value associated with the first choice alphas(Ei) and the index i.
'''

def selectJ(i,os,Ei):	# Inner loop heuristic
	maxK = -1 ; maxDeltaE =0; Ej = 0
	# Set the input Ei to valid in the cache. 
	os.eCache[i] = [1,Ei]
	# this create a list nonzeros values in eCache. The numpy function nonzero() returns a list containing not zero. 
	validEcacheList = nonzero(os.eCache[:,0].A)[0]
	
	# Loop through all values to check which one gives you max change
	if (len(validEcacheList)) > 1:
		for k in validEcacheList:
			if k == i:
				continue
			Ek = calcEk(os,k)
			deltaE = abs(Ei - Ek)
			
			if (deltaE > maxDeltaE):
					maxK =k 
					maxDelta = deltaE
					Ej = Ek
		return maxK,Ej
	else:
		# If this is the only value, then chose randomly
		j = selectJrand(i,os.m)
		Ej = calcEk(os,j)
	return j,Ej
			

''' calculates the error and puts it in the cache'''

def updateEk(os,k):
	Ek = calc(os,k)
	os.eCache[k] = [1,Ek]
	
######################################################### Full Platt SMO optimization routine #################################

''' Same code like simplified SMO , but in different format '''

def innerL(i,os):
	Ei = calcEk(os,i)
	if ((os.labelMat[i]*Ei < os.tol) and ( os.alphas[i] < os.C)) or (os.labelMat[i] * Ei > os.tol and (os.alpha[i] > 0)):
		# 
		j,Ej = selectJ(i,os,Ei)
		alphasIold = os.alpha[i].copy()
		alphasJold = os.alpha[j].copy()
		
		if (os.labelMat[i] != os.labelMat[j]):
			L = max(0,os.alphas[j] - os.alpha[i])
			H = min(os.C, os.C + os.alphas[j] - os.alpha[i])
		else:
			L = max(0,os.alphas[j] + os.alpha[i] - os.C)
			H = min(os.C, os.alphas[j] + os.alpha[i])
		
		if L==H:
			print "L==H"
			return 0
		
		eta = 2.0 * os.X[i,:] * os.X[j,:].T - os.X[i,:] * os.X[i,:].T - os.X[j,:] * os.X[j,:].T 
		if eta >= 0:
			print "eta >=0"
			return 0
		
		os.alphas[j] -= os.labelMat[j] * (Ei - Ej) / eta
		os.alphas[j]  = clipAlpha(os.alpha[j],H,L)
		updateEk(os,j) 		# Updates Ecache
	
		b1 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.X[i,:]*os.X[i,:].T - os.labelMat[j] * (os.alpha[j]-alphaJold)*os.X[i,:]*os.X[j,:].T
		b2 = oS.b - Ej- os.labelMat[i]*(os.alphas[i]-alphaIold)* os.X[i,:] * os.X[j,:].T - os.labelMat[j] * (os.alpha[j]-alphaJold)*os.X[j,:]*os.X[j,:].T
		
		if ( 0 < os.alphas[i]) and (os.C > os.alpha[i]):
			os.b = b1
		elif (0 < os.alphas[j]) and (os.C > os.alpha[j]):
			os.b = b2
		else:
			os.b = (b1+b2) / 2.0
		return 1
	else:
		return 0
	
		

	

######################################################### Full Platt SMO outer loop ############################################

''' This is the outer loop where you select your first alphas '''


def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin',0)):
	os = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
	iters = 0
	
	entireSet = True
	alphasPairsChanged = 0
	
	while ( iters < maxIter ) and (( alphasPairsChanged > 0 ) or (entireSet)):
		alphasPairsChanged = 0
		if entireSet:
			for i in range(os.m):
				# We call innerL() to choose a second alphas and do optimization if possible
				alphasPairsChanged += innerL(i,os)
				print "fullSet, iters: %d i:%d, pairs changed %d" % (iters,i,alphasPairsChanged)
				iters += 1
		else:
			nonBoundIs = nonzero((os.alphas.A > 0) * (os.alpha.A < C)) [0]
			# This for loop goes over all non-bound alphas , the values that aren't bound at 0 and C
			for i in nonBoundIs:
				alphasPairsChanged += innerL(i,os)
				print "non-bound, iter: %d i:%d, pairs changed %d" % (iters,i,alphasPairsChanged)
			iters += 1
		

		if entireSet:
			entireSet = False
		elif (alphasPairsChanged ==0 ):
			entireSet = True
		print "Iteration number : %d iterations \n" % iters
	return os.b, os.alphas






##################################################### Kernel Transformation function ################################################

''' KTup contains the information about the kernel, KernelTrans() takes 3 numeric types and a tuple , tuple holds the information about the kernel
    First  argument in the tuple holds information about that kernel ,  First agument decides which type of kernel it should use
    The function first create a column vector and then checks the tuple to check which type of kernel is being evaluated. 


    In case of linear kernel, a dot product is taken between the two inputs, which are 
    '''

def kernelTrans(X,A,kTup):
	m,n = shape(X)
	K = mat(zeors((m,1)))
	if kTup[0]=='lin':
		K = X * A.T
	elif kTup[0] == 'rbf':
		for j in range(m):
			deltaRow = X[j,:] - A 
			K[j] = deltaRow * deltaRow.T 
		K = exp ( K /(-1 * kTup[1] ** 2))
	else: # Raise an Exception if we encounter a tuple which we dont recognize
		raise NameError('We have a problem , The kernel is not recognized')
	return K 			







				
