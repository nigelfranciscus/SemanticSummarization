import pymongo
import codecs
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import ngrams

stop = set(stopwords.words('english'))
result = set()
f = codecs.open('C:/Users/s2876731/Desktop/in.txt', 'w', "utf-8")
uri = 'mongodb://bigdata:databig@localhost/?authSource=admin'


def search(myDict, lookup):
    for key, value in myDict.items():
        for v in value:
            if lookup in v:
                return key


if __name__ == '__main__':
    db_connect = pymongo.MongoClient(uri)
    database_name = 'twitter'
    database = db_connect[database_name]
    collection = database.collection_names(include_system_collections=False)

stopWords = set(stopwords.words('english'))

for data in database['goldcoast_location'].find({}, {'text': 1, '_id': 0, 'lang': 1}).sort("_id", -1).limit(100):
    if data['lang'] == 'en':
        result = data["text"].replace("\n", " ")
        result = re.sub(r"https\S+", "", result)
        result = result.lower()
        resultsplit = result.split()

        stemmer = PorterStemmer()
        tokens = word_tokenize(result)
        tknzr = TweetTokenizer()
        tweettoken = tknzr.tokenize(result)
        tweetstemmer = stemmer.stem(tokens)
        print(tweetstemmer)
        # tagged = nltk.pos_tag(tokens)
        # print(tagged)
        threegrams = ngrams(tweettoken, 3)
        #for grams in threegrams:
            #print(grams)
