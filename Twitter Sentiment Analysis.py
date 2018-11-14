#SENTIMENT ANALYSIS
#first row of tweet
with open('pos_tweets.txt', 'r') as posfile, open('neg_tweets.txt', 'r') as negfile:
    print posfile.readline()
    print negfile.readline()


tweet = "I LOVE @Health4UandPets u guys r the best!! \""
usedwords = [word for word in tweet.split()]
print usedwords

#remove mention@
usedwords = [word for word in tweet.split() if word[0] !="@" ]
print usedwords

#remove words shorter than length=3 and http
import string
usedwords = [word for word in tweet.split() if word[0] !="@"  and len(word)>=3 and not word.startswith('http')]
print usedwords

#lower case and remove punctuation
import string
usedwords = [word.lower().translate(None,string.punctuation) for word in tweet.split() if word[0] !="@"  and len(word)>=3 and not word.startswith('http')]
print usedwords

#dictionary positive negative
tweets = []
with open('neg_tweets.txt','r') as infile:
    for line in infile:
        usedwords = [word.lower().translate(None,string.punctuation) for word in line.split() if word[0] !="@"  and len(word)>=3 and not word.startswith('http')]
        # Create a dictionary with the structure (word, True)
        dictwords = dict([(word,True) for word in usedwords])
        if len(dictwords) > 0: # We omit empty tweets
            tweets.append((dictwords,'negative'))
print usedwords
print dictwords
print tweets[-1]

#true means that a word is included in tweet
with open('pos_tweets.txt','r') as infile:
    for line in infile:
        usedwords = [word.lower().translate(None,string.punctuation) for word in line.split() if word[0] !="@"  and len(word)>=3 and not word.startswith('http')]    
        # Create a dictionary with the structure (word, True)
        dictwords = dict([(word,True) for word in usedwords])
        if len(dictwords) > 0: # We omit empty tweets
            tweets.append((dictwords,'positive'))
print usedwords
print dictwords
print tweets[-1]

#shuffle tweets
import random
random.shuffle(tweets)
print tweets[0]
print tweets[-1]

#tweets train-test #labels are included in tweets, NLKT will handle them
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(tweets,test_size=0.25)

print len(train_data)
print len(test_data)

#Naive bayes-nlkt classify
from nltk.classify import NaiveBayesClassifier, util
import time
start = time.time()
classifier= NaiveBayesClassifier.train(train_data)

print 'Elapsed time:', time.time() - start
print 'Obtained Accuracy:', util.accuracy(classifier,test_data)
classifier.show_most_informative_features(n=20)