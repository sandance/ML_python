

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	centroids = createCent(dataSet,k)
	clusterChanged = True

	while clusterChanged:
		clusterChanged = False
		for i in range(m): # For every row in the initial matrix
			minDist = inf ; minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI ; minIndex = j
			if clusterAssment[i,0] != minIndex:
					clusterChanged = True
			clusterAssment[i,:] = minIndex, minDist**2
		print centroids
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent ) [0]
			centroids[cent,:] = mean(ptsInClust, axis=0)
	return centroids, clusterAssment
