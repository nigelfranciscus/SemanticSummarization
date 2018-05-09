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
from sumy.summarizers.lsa import LsaSummarizer as LsaRank
from sumy.summarizers.text_rank import TextRankSummarizer as TextRank
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRank
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

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
for data in database['games_sample_100000_aggregation'].\
        find({}, {'_id': 0, 'text': 1, 'extended_tweet': 1, 'lang': 1, 'str_id': 1}).limit(1000):
    if data['lang'] == 'en':
        if 'extended_tweet' in data:
            data['text'] = data['extended_tweet']['full_text']

        result = data["text"].replace("\n", " ")
        result = result.replace("RT", "")
        result = re.sub(r"https\S+", "", result)
        result = re.sub('[^A-Za-z0-9]+', ' ', result).lower()

        # print("ID : %s \nOriginal Text: %s" % (data["id_str"], result))
        # output_tweet = (data["id_str"], result)
        # print(output_tweet[0])
        final_result.add(result)

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

LANGUAGE = "english"
SENTENCES_COUNT = 10

# url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
# parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
# or for plain text files
# TODO : Implement categories / paragraph vectoring beforehand.
join_result = ".\n".join(final_result)
# print(join_result)
parser = PlaintextParser.from_string(join_result, Tokenizer(LANGUAGE))
stemmer = Stemmer(LANGUAGE)

summarizer = LexRank(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)
