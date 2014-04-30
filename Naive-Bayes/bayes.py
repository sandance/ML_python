import numpy as np
from numpy import *
import math
from math import *

def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea','problems', 'help', 'please'],['maybe', 'not', 'take', 'him','to', 'dog', 'park', 'stupid'], \
			['my', 'dalmation', 'is', 'so', 'cute','I', 'love', 'him'],['stop', 'posting', 'stupid', 'worthless', 'garbage'], \
			['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1] #1 is abusive , 0 not
	return postingList,classVec


# This will create a list of all the unique words in all of our documents


def createVocabList(dataSet):
	vocabSet = set([])  				# empty set
	for documents in dataSet:
		vocabSet = vocabSet | set (documents)   # Create the union of two sets
	return list(vocabSet)


# This takes the vocabulary list and creates a output vector of 1 and 0 

def setOfWords2Vec(vocabList,inputSet):
	returnVec =[0]*len(vocabList)                         # create a vector the same length as the vocabulary list and fill up with 0's
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1    # if the word in the vocabulary list 
		else:
			print "The word: %s  is not in my vocabulary!" % word
	return returnVec




# bag of words impletementation. This can have multiple occurances of each word 

def bagOfWords2VecMN(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] +=1
	return returnVec




""" Pseucode for the next process 
	Count the number of documents in each class
	for every training document
		for each class 
			 if a token appears in the document -> increement the count for that token	
			 increment the count for tokens
		for each class
			divide the token count by the number of total count to get conditional probabilites
		return conditional probabilities for each class
"""


def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	p0Num    = np.ones(numWords);
	p1Num 	 = np.ones(numWords);
	p0Denom  =2.0
	p1Denom  =2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
				p1Num += trainMatrix[i]
				p1Denom += sum(trainMatrix[i])    # vector addition 
		else:
				p0Num += trainMatrix[i]
				p0Denom += sum(trainMatrix[i])
	#p1Vect = math.log(p1Num/p1Denom)
	#p0Vect = math.log(p0Num/p0Denom)
	p1Vect= np.log10(np.abs(p1Num/p1Denom))
	p0Vect= np.log10(np.abs(p0Num/p0Denom))
	return p0Vect, p1Vect,pAbusive


def classifyNB(vec2Classify, p0Vec , p1Vec, pClass1):
	p1=sum(vec2Classify * p1Vec) + math.log(pClass1)
	p0=sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
	print p0
	print p1
	if p1 > p0:
		return 1
	else:
		return 0
		

def testingNB():
	listOPosts, listClasses=loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat=list()
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	
	p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
	testEntry=['Love','my','dalmation']
	thisDoc  = np.array(setOfWords2Vec(myVocabList,testEntry))
	print testEntry, ' Classified as:  ',classifyNB(thisDoc,p0V,p1V,pAb)
	testEntry=['stupid','garbage']
	thisDoc  = np.array(setOfWords2Vec(myVocabList,testEntry))
        print testEntry, ' Classified as:  ',classifyNB(thisDoc,p0V,p1V,pAb)
	



#Cross validation with naive bayes

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
	docList=list(); classList=list(); fullText=list()
	for i in range(1,26):
		wordList=textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList=textParse(open('email/ham/%.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)
	trainingSet= range(50) ; testSet=list()
	#creating a test set, 10 of the emails are randomly selected  to be used in test set
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))  # randomly create the training set
		testSet.append(trainingSet[randIndex])
		del (trainingSet[randIndex])
	trainMat=list(); trainClasses=list()
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,doclist[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount=0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
				errorCount +=1
	print 'The error rate is:  ', float(errorCount)/ len(testSet)	



def calcMostFreq(vocabList,fullText):
	#Calculate frequency of occurance
	import operator
	freqDict = dict()
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq= sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedFreq

'''
def localWords(feed1,feed0):
	import feedparser
	docList,classList,fullText=list(),list(),list()
	minLen=min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		# access one feed at a time
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	#Remove most frequenty occuring words
	top30Words = calcMostFreqi(vocabList,fullText)	
	for pairW in top30Words:
		if pairW[0] in vocablist:
			vocabList.remove(pairW[0])
	trainingSet = range(2*minLen); testSet= list()
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del (trainingSet[randIndex])
	trainMat,trainClasses=list(),list()
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount =0
	for 
'''
