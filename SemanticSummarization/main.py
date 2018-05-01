import json
import random

import pymongo
import codecs
import re

from nltk.tag.stanford import CoreNLPPOSTagger
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import ngrams
from nltk import pos_tag
from nltk.corpus import brown

from sklearn.cluster import KMeans

from SemanticSummarization.twokenize import simpleTokenize
from SemanticSummarization.k_means import kMeans

stops = set(stopwords.words("english"))
stemmer = PorterStemmer()
# tknzr = TweetTokenizer()
# tokens = word_tokenize(result)

result = set()
final_result = set()
# f = codecs.open('C:/Users/s2876731/Desktop/in.txt', 'w', "utf-8")
uri = 'mongodb://bigdata:databig@localhost/?authSource=admin'



if __name__ == '__main__':
    db_connect = pymongo.MongoClient(uri)
    database_name = 'twitter'
    database = db_connect[database_name]
    collection = database.collection_names(include_system_collections=False)

for data in database['goldcoast_location'].find({}, {'text': 1, '_id': 0, 'id_str': 1, 'lang': 1}).sort("_id", -1).limit(10):
    if data['lang'] == 'en':
        result = data["text"].replace("\n", " ")
        result = re.sub(r"https\S+", "", result)
        result = re.sub('[^A-Za-z0-9]+', ' ', result).lower()
        # print(data)
        # print("ID : %s \nOriginal Text: %s" % (data["id_str"], result))
        output_tweet = (data["id_str"], result)
        # print(output_tweet[0])
        final_result.add(output_tweet)

        tweet_token = simpleTokenize(result)
        # Use the universal tagger
        tagged_token = pos_tag([w for w in tweet_token if w not in stops], tagset="universal")
        # print("Tag : %s" % tagged_token)
        # Only get specific POS (such as N (Noun), A (Adverb), V (Verb))
        stem_token = [stemmer.stem(tag) for tag in
                      [item[0] for item in tagged_token if item[1] == 'ADJ' or item[1] == 'VERB']]
        # print("Specific tag only: %s" % stem_token + "\n")

        three_grams = ngrams(tagged_token, 3)
        # for grams in three_grams:
        # print(grams)

tweets = {}
in_tweet = {}
for each_tweet in final_result:
    tweets['text'] = each_tweet[1]
    in_tweet[int(each_tweet[0])] = tweets

print(in_tweet.keys())

seeds = [991205551961853952, 991205975917936640]

kmeans = kMeans(seeds, in_tweet)
kmeans.converge()
kmeans.printClusterText()
kmeans.printClusters()