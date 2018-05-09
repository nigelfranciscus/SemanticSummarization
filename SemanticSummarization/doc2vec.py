from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import json
import random
import pymongo
import codecs
import re
import os
import smart_open

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

from collections import namedtuple

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim.models.doc2vec import TaggedDocument
from collections import OrderedDict
import multiprocessing

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
    # collection = database.collection_names(include_system_collections=False)

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

#print(len(final_result))
join_result = ".\n".join(final_result)
#print(join_result)

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(join_result)]
print(tagged_data)

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

model= Doc2Vec.load("d2v.model")
# to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
print(model.most_similar('1'))

similar_doc = model.docvecs.most_similar(('1'))
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])


# put the line identifier
# tweet_identifier = []
# for idx, line in enumerate(join_result.splitlines()):
#     num_line = u"_*{0} {1}\n".format(idx, line)
#     tweet_identifier.append(num_line)
#
# SentimentDocument = namedtuple('TweetDocument', 'words tags split tweets')
#
# alldocs = []  # Will hold all docs in original order
# for line_no, line in enumerate(tweet_identifier):
#     tokens = gensim.utils.to_unicode(line).split()
#     words = tokens[1:]
#     tags = [line_no]  # 'tags = [tokens[0]]' would also work at extra memory cost
#     split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
#     sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
#     alldocs.append(SentimentDocument(words, tags, split, sentiment))
#
# train_docs = [doc for doc in alldocs if doc.split == 'train']
# test_docs = [doc for doc in alldocs if doc.split == 'test']
# doc_list = alldocs[:]  # For reshuffling per pass
#
# print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))