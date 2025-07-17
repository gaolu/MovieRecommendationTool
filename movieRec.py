'''
This is the Python program for the final project of
ECEN 689: Advanced Computational Methods for Integrated System Design.

This program takes movie and user information from GroupLens database
and IMDb database. The program aims in providing friends and movie
recommendations to existing users.

This program utilizes the following machine learning algorithms:

Vector Space Model (VSM)
K-Means Clustering
Naive Bayes Classification
Association Rules Learning

Author: Lu Gao
Email: gaolu.adam@gmail.com
Date of Last Modified: 04-29-2013
'''

'''test comment'''

# import moduels
import math
import random
import csv
import json
import re
import os.path
import sys

from operator import itemgetter
from urllib2 import Request, urlopen
from itertools import chain, combinations
from collections import defaultdict

# global variables
max_clusters = {} # store cluster information which reaches max RSS value (best similarity)
max_RSS = 0.0 # store the max RSS value
newMovieList = {} # store movies that have not been watched by existing GroupLens users
movieDict = {} # store the new movie ratings from IMDb

def getUserRatings(filename):
    '''
    Get movie ratings from existing GroupLens users
    '''
    # format user:{movie1: rating1, movie2: rating2,...}
    userRatings = {}
    f = file(filename, "r")
    lines = f.readlines()
    for line in lines:
        movieRating = {} # temp dict {movie1: rating1, movie2: rating2,...}
        strList = line.rstrip().split('	')
        userID = strList[0]
        movieID = strList[1]
        rating = strList[2]
        movieRating[movieID] = rating
        if userID not in userRatings: # first time appearance
            userRatings[userID] = movieRating
        else:
            userRatings[userID][movieID] = rating
    f.close()
    return userRatings

def tfIdfVectorizer(userRatingDict):
    '''
    Transform the ratings to tf-idf vectorized form and do normalization
    '''
    # the dict has already got raw term frequencies (tf) indicated by their ratings
    # calculate the document frequency (df)
    docFreq = {}
    for user in userRatingDict:
        for movie in userRatingDict[user]:
            if movie not in docFreq:
                docFreq[movie] = 1
            else:
                docFreq[movie] = docFreq[movie] + 1
    N = len(userRatingDict)
    # combine tf and df then do tf-idf (inverted df)
    for user in userRatingDict:
        for movie in userRatingDict[user]:
            userRatingDict[user][movie] = (math.log(float(userRatingDict[user][movie]), 2) + 1) * (math.log(N, 2) - math.log(docFreq[movie], 2))
    # do normalization
    for user in userRatingDict:
        sum = 0.0
        for movie in userRatingDict[user]:
            sum = sum + userRatingDict[user][movie] * userRatingDict[user][movie]
        sum = math.sqrt(sum)
        if sum != 0.0:
            for movie in userRatingDict[user]:
                userRatingDict[user][movie] = userRatingDict[user][movie] / sum
        if sum == 0.0: # if never rated movie
            for movie in userRatingDict[user]:
                # smoothing with 1e-8
                userRatingDict[user][movie] = 0.00000001
    return userRatingDict

def getDistance(userRatingDict):
    '''
    Calculate the distances between two users based on their tf-idf vectors
    '''
    # construct a dictionary that indicate the similarity between each pair of documents
    # formatc: user1:{user2: distance1, user3: distance2,...}
    distanceDict = {}
    for currUser in userRatingDict:
        distanceMap = {}
        for otherUser in userRatingDict:
            # get the distance between two documents
            distance = distanceCalculator(userRatingDict[currUser], userRatingDict[otherUser])
            distanceMap[otherUser] = distance
        distanceDict[currUser] = distanceMap
    return distanceDict
    
def distanceCalculator(dict1, dict2):
    '''
    Distance calculator between any two vectors
    '''
    # calculate the distance between two dictinaries: word: weight
    distance = 0
    for key in dict1:
        if key in dict2:
            value = dict1[key] * dict2[key]
            distance = distance + value
    return distance
    
def getCluster(distanceDict, k):
    '''
    CORE FUNCTION
    cluster nodes into k clusters based on the distanceDict that contains the distance between any two nodes
    the function runs 100 times k-means algorithm and return the clusters that has maximum euclidean value (best similarity)
    '''
    global max_RSS # actually means euclidean similarity in this algorithm, the large the better
    global max_clusters
    # run k-means 100 times and return the best value (largest euclidean value)
    for x in range(100):
        # sometimes the random picking will generate some identical pickups, need to avoid that
        #print x
        key_OK = False
        while key_OK != True:
            initCenter = []
            for i in range(k):
                # randomly picking starting nodes
                initCenter.append(random.choice(distanceDict.keys()))
            if len(list(set(initCenter))) == k:
                key_OK = True
        converge = False
        maxIter = 20 # maximum iteration number
        iterNumber = 0
        
        # start k-means algorithm until convergence
        final_RSS = 0.0
        while True:
            clusters = {}
            for ID in distanceDict[initCenter[0]]:
                if ID in initCenter:
                    continue
                local_max = 0
                max_ID = ""
                for i in range(k):
                    value = distanceDict[initCenter[i]][ID]
                    # compare the node distance to each center and store the shortest (biggest euclidean value) center to that node
                    if value >= local_max:
                        local_max = value
                        max_ID = initCenter[i]
                        # update the center of that node
                        clusters[ID] = initCenter[i]
            prev_center = initCenter # store the previous iteration centers
            next_center = []
            RSS = 0.0 # initialize the RSS to be 0 and RSS will increase with iteration
            for i in range(k):
                nodelist = []
                for key in clusters:
                    if clusters[key] == initCenter[i]:
                        nodelist.append(key)
                nodelist.append(initCenter[i])
                subDistance = {}
                for curr_node in nodelist:
                    distanceSum = 0.0
                    for other_node in nodelist:
                        # sum of distance of nodes in a cluster to its center
                        distanceSum = distanceSum + distanceDict[curr_node][other_node]
                    subDistance[curr_node] = distanceSum - distanceDict[curr_node][curr_node]
                # calcualte present euclidean value
                next_center.append(keyMaxValue(subDistance))
                RSSValue = subDistance[keyMaxValue(subDistance)]/float(len(nodelist))
                RSS = RSS + subDistance[keyMaxValue(subDistance)]
                if RSS > final_RSS:
                    final_RSS = RSS
            iterNumber = iterNumber + 1
            # converged as the the centers in two iterations are the same or reach the max iteration number
            if next_center == prev_center or iterNumber >= maxIter:
                initCenter = next_center
                break
            # not converged pass the previous centers to the k-means algorithm for another round of selection
            initCenter = next_center
        # select the biggest similarity cluster
        if final_RSS > max_RSS:
            max_RSS = final_RSS
            max_clusters = clusters
    # print max_RSS # may de-comment to print out the RSS value    
    return max_clusters

def keyMaxValue(dict):
    '''
    Get the key with max value
    '''
    # return the key of a dictionary that contains biggest value
    v = list(dict.values())
    k = list(dict.keys())
    return k[v.index(max(v))]
    
def clusterToGroup(clusters):
    '''
    Group clusters together and fine print
    '''
	# group each cluster together and finer print out
    # clusters dict format: node1: center1, node2: center5, node3: center2,...
    nodeList = list(clusters.values())
    nodeList = list(set(nodeList))
    returnList = []
    n = 0
    for node in nodeList:
		clusterList = []
		for key in clusters:
			if clusters[key] == node:
				clusterList.append(key)
		clusterList.append(node)
		print "cluster " + str(n)
		n = n + 1
		print clusterList
		print '\n'
		print "========================================="
		print '\n'
		returnList.append(clusterList)
    return returnList

def getPurity(clusterList):
    '''
    Calculate purity of clustering algorithm, not used in this program but saved for later
    '''
    return None

def getRI(clusterList):
    '''
    Calculate Rand Index (RI) of clustering algorithm, not used in this program but saved for later
    '''
    return None

def getGenreList(filename):
	# get a list of genres
	genreList = []
	f = file(filename,"r")
	lines = f.readlines()
	for line in lines:
		itemList = line.rstrip().split('|')
		genreList.append(itemList[0])
	return genreList

def getMovieGenre(filename, genreList):
	# relate each movie with their genres
    # format: movie1: genre1, movie2: genre2,...
	movieGenre = {}
	f = file(filename, "r")
	lines = f.readlines()
	for line in lines:
        # parse the line to get movieID and genres
		wordList = line.rstrip().split('|')
		movieID = wordList[0]
		wordList = wordList[-19:]
		movieGenreList = []
		indices = [i for i, x in enumerate(wordList) if x == "1"]
		for index in indices:
			movieGenreList.append(genreList[index])
		movieGenre[movieID] = movieGenreList
	return movieGenre

def getUserGenre(filename, movieGenre):
    '''
    Relate the user's movie rating with corresponding genres
    '''
	# user:{genre1: rating1, genre2: rating2,...}
    userGenres = {}
    f = file(filename, "r")
    lines = f.readlines()
    for line in lines:
        genreRating = {}
        strList = line.rstrip().split('	')
        userID = strList[0]
        movieID = strList[1]
        rating = strList[2]
        genreList = movieGenre[movieID]
        for genre in genreList:
        	genreRating[genre] = int(rating) # integerize
        for genre in genreList:
        	if userID not in userGenres:
        		userGenres[userID] = genreRating
        	else:
        		if genre in userGenres[userID]:
                    # sum up the genres ratings
        			userGenres[userID][genre] = userGenres[userID][genre] + genreRating[genre]
        		else:
        			userGenres[userID][genre] = genreRating[genre]
    f.close()
    return userGenres

def combineDistance(distanceDict, genreDistanceDict):
    '''
    Combine the distance data generated from movie vector and genre vector
    '''
    for userID in distanceDict:
		for otherUser in distanceDict[userID]:
			distanceDict[userID][otherUser] = 0.75 * distanceDict[userID][otherUser] + 0.25 * genreDistanceDict[userID][otherUser]
    return distanceDict

def getFriends(clusterList, distanceDict):
    '''
    CORE FUNCTION TO RECOMMEND FRIENDS
    Get top five nearest users in the same cluster
    '''
    N = 5
    userFriends = {}
    for cluster in clusterList:
        smallDistanceDict = {}
        for currUser in cluster:
            distance = {}
            for otherUser in cluster:
                # disregard the self-to-self distance
                if currUser == otherUser:
                    continue
                distance[otherUser] = distanceDict[currUser][otherUser]
            # get the nearset 5 users in the same cluster
            friendsList = getTopKeys(distance, N)
            userFriends[currUser] = friendsList
    # fine print
    for user in userFriends:
        print "user: ", user, "  recommended friends: ", userFriends[user]
    return userFriends
                
def getTopKeys(dict, k):
    # get the top k keys with their value larger than the rest of the element's values
    return map(itemgetter(0), sorted(dict.items(), key=itemgetter(1), reverse=True))[:k]
    
def getMovieNames(filename):
    '''
    Get movie names to associate with their movie ID number
    '''
    movieNameDict = {}
    f = file(filename, "r")
    lines = f.readlines()
    for line in lines:
        wordList = line.rstrip().split('|')
        movieID = wordList[0]
        movieName = wordList[1]
        movieNameDict[movieID] = movieName
    return movieNameDict
    
def getNewMovieRatings(moviefile, ratingfile):
    '''
    Get movie ratings from the IMDb database for movies not watched by the GroupLens users
    '''
    global newMovieList # use the global variable
    f = file(moviefile, "r")
    lines = f.readlines()
    n = 0
    for line in lines:
        movieYear = {}
        infoList = line.rstrip().split('\t')
        movieName = infoList[0]
        for year in infoList[-1:]:
            try:
                if int(year) >= 2009: # only new movies
                    newMovieList[movieName] = year
            except ValueError:
                pass
    f.close()
    
    global movieDict
    f = file(ratingfile, "r")
    lines = f.readlines()
    n = 0
    for line in lines:
        movieRate = {}
        infoList = line.rstrip().split('  ')
        n = n + 1
        for rating in infoList[-2:-1]:
            movieRating = float(rating)
        for name in infoList[-1:]:
            movieName = name
        if movieName in newMovieList:
            movieRate['rating'] = movieRating
            movieDict[movieName] = movieRate
    f.close()
    return movieDict
    
def getNewMovieGenres(genrefile):
    '''
    Get movie genres from the IMDb database for movies not watched by the GroupLens users
    '''
    global movieDict
    newMovieGenre = {}
    f = file(genrefile, "r")
    lines = f.readlines()
    for line in lines:
        infoList = line.rstrip().split('\t')
        movieName = infoList[0]
        movieGenre = infoList[-1]
        if movieName in movieDict:
            newMovieGenre[movieName] = movieGenre
    # genre list contains the common movies between rating list and genre list
    return newMovieGenre

def getNewMovieDirector(directorfile, newMovieGenre):
    '''
    Get movie directors from the IMDb database for movies not watched by the GroupLens users
    '''
    newMovieDirector = {}
    directorName = ''
    f = file(directorfile, "r")
    lines = f.readlines()
    for line in lines:
        infoList = line.rstrip().split('\t')
        newMovieName = infoList[-1]
        if infoList[0] != '':
            newMovieDirector[newMovieName] = infoList[0]
            directorName = infoList[0]
        else:
            newMovieDirector[newMovieName] = directorName
    f.close()
    return newMovieDirector

def getNewMovieActress(actressfile):
    '''
    Get movie actresses from the IMDb database for movies not watched by the GroupLens users
    '''
    # format: actress: movieList
    newMovieActress = {}
    actressName = ''
    lastActressName = ''
    movieList = []
    f = file(actressfile, "r")
    lines = f.readlines()
    for line in lines:
        infoList = line.rstrip().split('\t')
        movieName = infoList[-1].split(')')[0] + ')'
        if len(infoList) != 1:
            if infoList[0] != '':
                actressName = infoList[0]
                movieList.append(movieName)
            else:
                movieList.append(movieName)
        else:
            newMovieActress[actressName] = movieList
            movieList = []
    f.close()
    return newMovieActress

def getNewMovieActor(actorfile):
    '''
    Get movie actors from the IMDb database for movies not watched by the GroupLens users
    '''
    # same format as actress list
    newMovieActor = {}
    actorName = ''
    lastActressName = ''
    movieList = []
    f = file(actorfile, "r")
    lines = f.readlines()
    for line in lines:
        infoList = line.rstrip().split('\t')
        movieName = infoList[-1].split(')')[0] + ')'
        if len(infoList) != 1:
            if infoList[0] != '':
                actorName = infoList[0]
                movieList.append(movieName)
            else:
                movieList.append(movieName)
        else:
            newMovieActor[actorName] = movieList
            movieList = []
    f.close()    
    return newMovieActor
    
def getOldMovieDirector(movieNameDict, newMovieDirector):
    '''
    Get movie directors from the IMDb database for movies watched by the GroupLens users
    '''
    oldMovieDirectorDict = {}
    for movie in movieNameDict:
        movieName = movieNameDict[movie]
        # use try-except block because there maybe some movies in GroupLens database
        # but not in IMDb database or vice versa
        try:
            movieDirector = newMovieDirector[movieName]
            oldMovieDirectorDict[movieName] = movieDirector
        except KeyError:
            pass
    return oldMovieDirectorDict

def getOldMovieActress(movieNameDict, newMovieActress):
    '''
    Get movie actresses from the IMDb database for movies watched by the GroupLens users
    '''
    oldMovieActress = {}
    n = 0
    for movie in movieNameDict:
        movieName = movieNameDict[movie]
        actressList = []
        for actress in newMovieActress:
            movieList = newMovieActress[actress]
            if movieName in movieList:
                actressList.append(actress)
        if len(actressList) != 0:
            oldMovieActress[movieName] = actressList
        n = n + 1
    return oldMovieActress
    
def getOldMovieActor(movieNameDict, newMovieActor):
    '''
    Get movie actors from the IMDb database for movies watched by the GroupLens users
    '''
    oldMovieActor = {}
    n = 0
    for movie in movieNameDict:
        movieName = movieNameDict[movie]
        actorList = []
        for actor in newMovieActor:
            movieList = newMovieActor[actor]
            if movieName in movieList:
                actorList.append(actor)
        if len(actorList) != 0:
            oldMovieActor[movieName] = actorList
        n = n + 1
    return oldMovieActor

def classTopDirector(movieNameDict, clusterList, userRatings, oldMovieDirectorDict):
    '''
    Each cluster have a director list
    Calculate the likeness of each director by users in the cluster
    Form the tf-idf vector used as probabilities that a cluster likes a director
    To be called by the Naive Bayes training function
    '''
    clusterFavoriteDict = {}
    n = 0
    # get the movie ratings first
    for cluster in clusterList:
        clusterName = 'cluster_' + str(n)
        movieRatingSum = {}
        for user in cluster:
            for movie in userRatings[user]:
                movieName = movieNameDict[movie]
                if movie not in movieRatingSum:
                    movieRatingSum[movieName] = userRatings[user][movie]
                else:
                    movieRatingSum[movieName] = movieRatingSum[movieName] + userRatings[user][movie]
        clusterFavoriteDict[clusterName] = movieRatingSum
        n = n + 1
        
    returnDict = {}
    # associate director values with corresponding movie rating values
    for cluster in clusterFavoriteDict:
        directorDict = {}
        for movie in clusterFavoriteDict[cluster]:
            if movie in oldMovieDirectorDict:
                if oldMovieDirectorDict[movie] not in directorDict:
                    directorDict[oldMovieDirectorDict[movie]] = clusterFavoriteDict[cluster][movie]
                else:
                    directorDict[oldMovieDirectorDict[movie]] = directorDict[oldMovieDirectorDict[movie]] + clusterFavoriteDict[cluster][movie]
        returnDict[cluster] = directorDict
    
    # normalization so sum up to 1
    for cluster in returnDict:
        clusterSum = 0.0
        for director in returnDict[cluster]:
            clusterSum = clusterSum + returnDict[cluster][director] * returnDict[cluster][director]
        clusterSum = math.sqrt(clusterSum)
        for director in returnDict[cluster]:
            returnDict[cluster][director] = returnDict[cluster][director] / clusterSum
    
    '''
    # used to get the top rated directors by each cluster of users
    # not used in this version
    topDirectorDict = {}
    for cluster in returnDict:
        topDirectors = getTopKeys(returnDict[cluster], 10)
        topDirectorDict[cluster] = topDirectors
    '''
    
    return returnDict
    
def classTopGenre(movieNameDict, clusterList, userRatings, movieGenre):
    '''
    Each cluster have a genre list
    Calculate the likeness of each genre by users in the cluster
    Form the tf-idf vector used as probabilities that a cluster likes a genre
    To be called by the Naive Bayes training function
    '''
    classGenreDict = {}
    n = 0
    # get the genre rating values
    for cluster in clusterList:
        clusterName = "cluster_" + str(n)
        genreRatingSum = {}
        for user in cluster:
            for movie in userRatings[user]:
                for genre in movieGenre[movie]:
                    if genre not in genreRatingSum:
                        genreRatingSum[genre] = userRatings[user][movie]
                    else:
                        genreRatingSum[genre] = genreRatingSum[genre] + userRatings[user][movie]
        classGenreDict[clusterName] = genreRatingSum
        n = n + 1
    
    # do normalization so sum up to 1
    for cluster in classGenreDict:
        clusterSum = 0.0
        for genre in classGenreDict[cluster]:
            clusterSum = clusterSum + classGenreDict[cluster][genre] * classGenreDict[cluster][genre]
        clusterSum = math.sqrt(clusterSum)
        for genre in classGenreDict[cluster]:
            classGenreDict[cluster][genre] = classGenreDict[cluster][genre] / clusterSum
    
    '''
    # used to get the top rated genres by each cluster of users
    # not used in this version
    topGenreDict = {}
    for cluster in classGenreDict:
        topGenres = getTopKeys(classGenreDict[cluster], 5)
        topGenreDict[cluster] = topGenres
    '''
    
    return classGenreDict
    
def naiveBayesTrain(movieNameDict, clusterList, userRatings, oldMovieDirectorDict, movieGenre):
    '''
    CORE FUNCTION TO TRAIN NAIVE BAYES CLASSIFIER
    '''
    # get director probabilities
    classTopDirectorDict = classTopDirector(movieNameDict, clusterList, userRatings, oldMovieDirectorDict)
    # get genre probabilities
    classTopGenreDict = classTopGenre(movieNameDict, clusterList, userRatings, movieGenre)
    
    return classTopDirectorDict, classTopGenreDict
    
def naiveBayesApply (classTopDirectorDict, classTopGenreDict, newMovieGenre, newMovieDirector, newMovieRatings):
    '''
    CORE FUNCTION TO CLASSIFY NEW MOVIES INTO CLUSTERS
    '''
    newMovieList = []
    for movie in newMovieGenre:
        if movie in newMovieDirector and newMovieRatings:
            newMovieList.append(movie)
    movieTargetClusterDict = {}
    for movie in newMovieList:
        director = newMovieDirector[movie]
        genre = newMovieGenre[movie]
        clusterValue = {}
        for cluster in classTopDirectorDict:
            clusterName = cluster
            directorValue = 0.0
            genreValue = 0.0
            if director in classTopDirectorDict[cluster]:
                # get director probability
                directorValue = classTopDirectorDict[cluster][director]
            if genre in classTopGenreDict[cluster]:
                # get genre probability
                genreValue = classTopGenreDict[cluster][genre]
            # calculate the probabilities that a movie belongs to each cluster
            totalValue = (directorValue * genreValue) * float(newMovieRatings[movie]['rating'])
            clusterValue[clusterName] = totalValue
        # classify based on highest probability
        targetCluster = keyMaxValue(clusterValue)
        movieTargetClusterDict[movie] = targetCluster
    return movieTargetClusterDict

def displayMovieRec(naiveBayesResult, classTopDirectorDict, newMovieRatings):
    '''
    Turn the dictionary format into list format and fine print
    '''
    movieRecDict = {}
    for cluster in classTopDirectorDict:
        movieList = []
        clusterName = cluster
        for movie in naiveBayesResult:
            if naiveBayesResult[movie] == clusterName:
                movieList.append(movie)
        movieRecDict[cluster] = movieList
    for cluster in movieRecDict:
        print cluster, ":" # cluster name
        print '----------------------------'
        if len(movieRecDict[cluster]) > 50:
            tempMovieDict = {}
            movieList = movieRecDict[cluster]
            for movie in movieList:
                tempMovieDict[movie] = newMovieRatings[movie]['rating']
            topList = getTopKeys(tempMovieDict, 50)
            for movie in topList:
                print movie
        else:
            for movie in movieRecDict[cluster]:
                print movie
        print '============================='
        print '\n\n'
    return movieRecDict

def writeUserMovie(userRatings):
    '''
    Write user movie list to csv file to be used by association rules part
    '''
    f = file('userMovie.csv', "a")
    for user in userRatings:
        movieList = userRatings[user].keys()
        f.write(str(movieList) + '\n')
    f.close()
    return 0

'''
The Association Rules Part
Suggest old movies that has not been watched by an individual user
'''

def subsets(arr):
    '''
    Returns non empty subsets of arr
    '''
    return chain(*[combinations(arr,i + 1) for i,a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
	'''
    Calculates the support for items in the itemSet and returns a subset of the itemSet 
	each of whose elements satisfies the minimum support
    '''
	_itemSet = set()
	localSet = defaultdict(int)

	for item in itemSet:
		for transaction in transactionList:
			if item.issubset(transaction):
				freqSet[item] += 1
				localSet[item] += 1
	
	for item,count in localSet.items():
		support = float(count)/len(transactionList)
		
		if support >= minSupport:
			_itemSet.add(item)
	
	return _itemSet

def joinSet(itemSet,length):
	'''
    Join a set with itself and returns the n-element itemsets
    '''
	return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList	= list()
    itemSet = set()
    for record in data_iterator:
        #print record
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))		# Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    '''
    run the apriori algorithm. data_iter is a record iterator
    Return both: 
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    '''
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    
    freqSet = defaultdict(int)
    largeSet = dict()				# Global dictionary which stores (key=n-itemSets,value=support) which satisfy minSupport
    assocRules = dict()				# Dictionary which stores Association Rules
    
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    
    currentLSet	= oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet,k)
        currentCSet = returnItemsWithMinSupport(currentLSet, transactionList, minSupport, freqSet)
        currentLSet = currentCSet
        k = k + 1
        
    # nested function declaration
    def getSupport(item):
            '''
            Local function which returns the support of an item
            '''
            return float(freqSet[item])/len(transactionList)

    toRetItems=[]
    for key,value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) 
                           for item in value])

    toRetRules=[]
    for key,value in largeSet.items()[1:]:
        for item in value:
            _subsets = map(frozenset,[x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain)>0:
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element),tuple(remain)), 
                                           confidence))
    return toRetItems, toRetRules

def printResults(items, rules):
    '''
    Prints the generated itemsets and the confidence rules
    '''
    for item, support in items:
        print "item: %s , %.3f" % (str(item), support)
    print "\n------------------------ RULES:"
    for rule, confidence in rules:
        pre, post = rule
        print "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)

def getAssRuleRec(items, rules, userRatings, movieNameDict):
    '''
    Get movie suggested by association rules
    '''
    userRecList = {}
    for user in userRatings:
        recList = []
        movieList = []
        for movie in userRatings[user]:
            # add '' to left and right of movie name to match the format
            movieList.append('\'' + movie + '\'')
        for rule, confidence in rules:
            pre, post = rule
            preLen = len(pre)
            postLen = len(post)
            n = 0
            for movie in pre:
                if movie.lstrip().rstrip() in movieList:
                    n = n + 1
            if n == preLen:
                for assoMovie in post:
                    # associated movie not watched by the user
                    if assoMovie not in movieList:
                        recList.append(assoMovie.lstrip().replace('\'', ''))
        if len(recList) != 0:
            userRecList[user] = list(set(recList))
    userMovieList = {}
    for user in userRecList:
        userMovie = []
        for movieID in userRecList[user]:
            movieName = movieNameDict[movieID]
            userMovie.append(movieName)
        userMovieList[user] = userMovie
    return userMovieList

def dataFromFile(fname):
	'''
    Function which reads from the file and yields a generator
    '''
	file_iter = open(fname, 'rU')
   	for line in file_iter:
		line = line.strip().rstrip(',') # Remove trailing comma
		record = frozenset(line.split(','))
		yield record

'''
Main Function
'''
if __name__ == '__main__':

    # get users ratings on movies
    userRatings = getUserRatings('u.data')
    if not os.path.isfile('userMovie.csv'):
        print writeUserMovie(userRatings)
    print "done getting user ratings"
    
    userRatingDict = tfIdfVectorizer(userRatings)
    print "done doing tf-idf"
    
    print "calculating node distances..."
    distanceDict = getDistance(userRatingDict)
    print "done calculating node distances"
    
    #purity = getPurity(clusterList)
    #RI = getRI(clusterList)
    
    genreList = getGenreList('u.genre')
    print "got genre list"
    
    movieGenre = getMovieGenre('u.item', genreList)
    print "done linking movie to their genres"
    
    userGenres = getUserGenre('u.data', movieGenre)
    print "got users ratings on genres"
    
    userGenreDict = tfIdfVectorizer(userGenres)
    print "done doing tf-idf based on genres"
    
    print "calculating node distances based on genres..."
    genreDistanceDict = getDistance(userGenreDict)
    print "done calculating node distances based on genres"

    combinedDistanceDict = combineDistance(distanceDict, genreDistanceDict)
    print "done calculating combined node distances"
    print "doing clustering, please wait..."
    print "\n"
    
    k = 31
    clusters = getCluster(combinedDistanceDict, k)

    clusterList = clusterToGroup(clusters)
    print "done doing clustering"
    
    userFriends = getFriends(clusterList, combinedDistanceDict)
    print "done getting friends for users"
    
    movieNameDict = getMovieNames('u.item')
    print "done getting movie names"


    # The Association Rules Part
    print "calcualting association rules..."
    inFile = dataFromFile('userMovie.csv')
    minSupport = 0.2
    minConfidence = 0.95
    items, rules = runApriori(inFile, minSupport, minConfidence)
    assRuleRecDict = getAssRuleRec(items, rules, userRatings, movieNameDict)
    
    print "users", '\t', "recommended movies"
    for user in assRuleRecDict:
        print user, '\t', assRuleRecDict[user]
    # done the Association Rules part
    
    
    print "getting new movie information..."
    newMovieRatings = getNewMovieRatings('movies.txt', 'ratings.txt')
    print "done getting new movie ratings"
    
    newMovieGenre = getNewMovieGenres('genres.txt')
    print "done getting new movie genres"

    newMovieDirector = getNewMovieDirector('directors.list', newMovieGenre)
    print "done getting new movie directors"
    
    '''
    # to be used in later versions
    newMovieActress = getNewMovieActress('actresses.list')
    print "The length of newMovieActress is: ", len(newMovieActress)
    
    newMovieActor = getNewMovieActor('actors.list')
    print "The length of newMovieActor is: ", len(newMovieActor)
    '''
    
    oldMovieDirectorDict = getOldMovieDirector(movieNameDict, newMovieDirector)
    print "done getting old movie directors"
    
    print "training naive bayes classifier..."
    classTopDirectorDict, classTopGenreDict = naiveBayesTrain(movieNameDict, clusterList, userRatings, oldMovieDirectorDict, movieGenre)
    
    print "done training, now applying classifier to classify movies into clusters"
    naiveBayesResult = naiveBayesApply (classTopDirectorDict, classTopGenreDict, newMovieGenre, newMovieDirector, newMovieRatings)
    
    movieRecDict = displayMovieRec(naiveBayesResult, classTopDirectorDict, newMovieRatings)
    
    '''
    # to be used in later versions
    oldMovieActress = getOldMovieActress(movieNameDict, newMovieActress)
    #print oldMovieActress
    print "The length of oldMovieActress is: ", len(oldMovieActress)
    
    oldMovieActor = getOldMovieActor(movieNameDict, newMovieActor)
    print "The length of oldMovieActor is: ", len(oldMovieActor)
    '''