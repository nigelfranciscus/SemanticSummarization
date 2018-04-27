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

from SemanticSummarization.twokenize import simpleTokenize

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
        #result = re.sub(r'\s+', '\s', result)
        result = re.sub('[^A-Za-z0-9]+', ' ', result)
        result = result.lower()
        #print(result)
        # resultsplit = result.split()

        tweettoken = simpleTokenize(result)
        filteredtoken = [w for w in tweettoken if not w in stops]


        # Use the universal tagger
        taggedtoken = pos_tag(filteredtoken, tagset="universal")
        print(taggedtoken)
        # Only get specific POS (such as N (Noun), A (Adverb), V (Verb))
        specific_tag = [item[0] for item in taggedtoken if item[1] == 'ADJ' or item[1] == 'VERB']
        stemtoken = [stemmer.stem(tag) for tag in specific_tag]
        print(stemtoken)


        threegrams = ngrams(taggedtoken, 3)
        # for grams in threegrams:
        # print(grams)
