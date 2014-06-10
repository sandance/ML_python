from numpy import *

def loadDataSet(fileName):
	dataMat = list()
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataMat.append(fltLine)
	return dataMat
	

''' Calculating euclidian distance '''

def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB,2)))


''' Selecting random centroid '''
'''
def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))
	# Random centroids are need to be within the bounds of the dataSet 
	# This is accomplished by finding the minimum and maximum values of each dimention in the dataSet
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j] - minJ)
		centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
	return centroids


'''

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids



'''
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2))) # First column for storing index of the cluster , 2nd one for storing the error
	centriods = createCent(dataSet,k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m): #Iterating over all row of the dataSet
			minDist = inf ; minIndex = -1
			for j in range(k): # For each centriod among all centroid 	
				distJI = distMeas(centroids[j,:], dataSet[i,:]) # calculate euclidian dist bet centroid and each row in dataSet
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment [i,:] != minIndex:
					clusterChanged = True
			custerAssment[i,:] = minIndex, minDist**2
		print centroids 
		# update centroid location
		for cent in range(k):
			#First , you are doing some array filtering to get all the points in a given cluster
			# next you are taking all mean 
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
			centroids[cent,:] = mean(ptsInClust,axis=0)
	return centroids,clusterAssment
'''

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
	    #print clusterAssment,i
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
	print "Centroids\n"
        print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment


########################## Improving cluster performance with postprocessing ###############

''' Bisecting K- means

Pseudocode:

	Start with all the points in one cluster

	while the number of clusters is less than k
	
		for every cluster
			
			measure total error
				Perform k-means clustering with k=2 on the given cluster

				measure total error after k- means has split the cluster in two

		choose the cluster  split that gives the lowest error and commit this split

'''



##########################  Bisecting k-means clustering algorithm ######################
'''
def biKmeans(dataSet, k , distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	# initially create one cluster	
	centroid0 = mean(dataSet, axis=0).tolist() [0]
	centList = [centroid0]

	# All the point in one cluster
	for j in range(m):
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:]) ** 2
		
	while (len(centList) < k):
			lowestSSE = inf
			for i  in range(len(centList)): # looping through all clusters
					ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i) [0] ,:]
					centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
					sseSplit = sum(splitClustAss[:,1])
					sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i) [0] ,1])
					print "seeSplit and notSplit : " , sseSplit,sseNotSplit
					if ( sseSplit + sseNotSplit) < lowestSSE:
						bestCentToSplit = i	
						bestNewCents = centroidMat
						bestClustAss = splitClustAss.copy()
						lowestSSE = sseSplit + sseNotSplit
			bestClustAss[nonzero(bestClustAss[:,0].A ==1) [0], 0] = len(centList)
			bestClustAss[nonzero(bestClustAss[:,0].A ==0) [0], 0] = bestCentToSplit
			print 'The bestCentTosplit is ', bestCentToSplit
			print 'len of bestClustAss is: ', len(bestClustAss)
			centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # replace a centroid with two best centroids
			centList.append(bestNewCents[1,:])
			clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
	return mat(centList,clusterAssment)

'''


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

##################################### displaying ####################################################################
import matplotlib
import matplotlib.pyplot as plt
import sys

if __name__=='__main__':
	filename=sys.argv[1]
	num_of_cluster=int(sys.argv[2])
	#datMat=mat(loadDataSet('testSet.txt'))
	print filename,num_of_cluster
	datMat=mat(loadDataSet(filename))
	myCentroids, clustAssing= biKmeans(datMat,num_of_cluster)
	fig =plt.figure()
	rect=[0.1,0.1,0.8,0.8]
        scatterMarkers=['s', 'o', '^','8','d', 'v', 'h', '>', '<']
	colorMarkers=['blue','red','black','green','yellow','magenta','cyan','white']
	axprops = dict(xticks=[], yticks=[])
	ax0=fig.add_axes(rect, label='ax0', **axprops)
	for i in range(num_of_cluster):
		ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
	        markerStyle = scatterMarkers[i % len(scatterMarkers)]
		colorStyle = colorMarkers[i % len(colorMarkers)]
		ax0.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle,color=colorStyle,s=90)
	ax0.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
	plt.show()



################################################### PlaceFinder API ##################################################

import urllib
import json

def geoGrab(stAddress, city):
	apiStem = 'http://where.yahooapis.com/geocode?'
	params  = {}
	params['flags'] = 'J'
	#params['appid'] = '4dPXss6q'
	params['appid'] = 'ppp68N8t'
	params['location'] = '%s %s' % (stAddress, city)
	url_params = urllib.urlencode(params)
	yahooApi = apiStem + url_params
	print yahooApi
	c = urllib.urlopen(yahooApi)
	return json.loads(c.read())


#from time 
			
