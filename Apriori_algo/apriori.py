'''
For each transaction in tran the dataSet:
	For each candidate itemset, can:
		Check to see if can is the subset of tran
		If so increment the count of can
	For each candidate itemset:
		If the support meets the minimum , keep this item
		Return list of frequenct itemset
'''

####################################  Generating Candidate Itemsets  ################################## 

def loadDataSet():
	return [[1,3,4] , [2,3,5] , [1,2,3,5] ,[2,5]]

def createC1(dataSet):
	C1 = list()
	for transaction in dataSet:
		for item in transaction:
			if not [ item ] in C1:
				C1.append([item])
	print "unsorted %s" % C1
	C1.sort()
	print C1
	return map(frozenset,C1)


def scanD (D, Ck, minSupport):
	ssCnt = dict()
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not ssCnt.has_key(can):
					ssCnt[can] = 1
				else:
					ssCnt[can] +=1
	numItems = float(len(D))
	retList = []
	supportData = dict()
	for key in ssCnt:
		support = ssCnt[key] / numItems
		if support >= minSupport:
			retList.insert(0,key)
		supportData[key] = support
	return retList,supportData



# This create Ck list
def aprioriGen(Lk, k):
	retList = []
	lenLK= len(Lk)
	for i in range(lenLk):
		for j in range(i+1, lenLk):
			L1 = list(Lk[i]) [:k-2]
			L2 = list(Lk[j]) [:k-2]
			print L1,L2
			L1.sort(); L2.sort()
			
			if L1==L2:
				retList.append(Lk[i] | Lk[j])
	return retList


#Full Apriori algorithm 
def apriori(dataSet,minSupport =0.5):
	C1 = createC1(dataSet)
	D  = map(set,dataSet)
	L1, supportData = scanD(D,C1,minSupport)
	L = [L1]
	k= 2
	while(len(L[k-2]) > 0):
		Ck= aprioriGen(L[k-2], k)
		print Ck
		Lk, supk = scanD(D,Ck,minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L, supportData

