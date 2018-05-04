from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

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

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer as TextRank
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRank
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

stops = set(stopwords.words("english"))
stemmer = PorterStemmer()
# tknzr = TweetTokenizer()
# tokens = word_tokenize(result)

# f = codecs.open('C:/Users/s2876731/Desktop/in.txt', 'w', "utf-8")
uri = 'mongodb://bigdata:databig@localhost/?authSource=admin'


if __name__ == '__main__':
    db_connect = pymongo.MongoClient(uri)
    database_name = 'twitter'
    database = db_connect[database_name]
    collection = database.collection_names(include_system_collections=False)

final_result = set()
for data in database['goldcoast_location'].find({}, {'text': 1, '_id': 0, 'id_str': 1, 'lang': 1}).limit(10):
    if data['lang'] == 'en':
        result = data["text"].replace("\n", " ")
        result = re.sub(r"https\S+", "", result)
        result = re.sub('[^A-Za-z0-9]+', ' ', result).lower()
        # print(data)
        # print("ID : %s \nOriginal Text: %s" % (data["id_str"], result))
        # output_tweet = (data["id_str"], result)
        # print(output_tweet[0])
        final_result.add(result)

        tweet_token = simpleTokenize(result)

join_result = ".\n".join(final_result)
# print(join_result)

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # Will hold all docs in original order
for line_no, line in enumerate(join_result):
    tokens = gensim.utils.to_unicode(line).split()
    words = tokens[1:]
    tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
    split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
    sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
    alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # For reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))