import pymongo
import codecs
import re
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import ngrams
from nltk import pos_tag
from nltk.corpus import brown

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

stops = set(stopwords.words("english"))
stemmer = PorterStemmer()
tknzr = TweetTokenizer()
# tokens = word_tokenize(result)

for data in database['goldcoast_location'].find({}, {'text': 1, '_id': 0, 'lang': 1}).sort("_id", -1).limit(10):
    if data['lang'] == 'en':
        result = data["text"].replace("\n", " ")
        result = re.sub(r"https\S+", "", result)
        result = result.lower()
        # resultsplit = result.split()

        tweettoken = tknzr.tokenize(result)
        filteredtoken = [w for w in tweettoken if not w in stops]
        stemtoken = [stemmer.stem(t) for t in filteredtoken]
        # print(brown.tagged_words(tagset="universal"))

        taggedtoken = pos_tag(stemtoken, tagset="universal")
        #print(taggedtoken)

        threegrams = ngrams(taggedtoken, 3)
        for grams in threegrams:
            print(grams)

