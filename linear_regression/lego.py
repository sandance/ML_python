''' Using regression to predict the price of a LEGO Set

1. Collect : Collect from Google Shopping API

2. Prepare: Extract price data from the returned JSON

3. Analyze : visually inspect the data

4. Train: We will build different models with stagewise linear regression 
          and straight forward linear regression .

5. Test: We will use cross validation to test the different models to see which one performs the best.

6. Use: The resulting model will be the object of this exercise

'''

from time import sleep
import json
import urllib2

def searchForSet (retX,retY, setNum, yr, numPce, origPrc):
	sleep(10)
	myAPIstr = 'get from code.google.com'
	searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products? \
			key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr,setNum)
	pg = urllib2.urlopen(searchURL)
	retDict = json.loads(pg.read())

	for i in range(len(retDict['items'])):
		try:
			currItem = retDict['items'][i]
			if currItem['product']['condition'] == 'new':
				newFlag = 1
			else:
				newFlag = 0
			
			listOfInv = currItem['product']['inventories']

			for item in listOfInv:
				sellingPrice = item['price']
				if sellingPrice > origPric * 0.5:
					print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPric,sellingPrice)
					retX.append([yr,numPce,newFlag,origPrc])
					retY.append(sellingPrice)
		except:
			print 'Problem with item %d' % i


def setDataCollect(retX, retY):
	searchForSet(retX, retY, 8288, 2006, 800, 49.99)
	searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
	searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
	searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
	searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
	searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

