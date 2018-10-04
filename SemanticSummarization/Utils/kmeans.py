import codecs

from gensim import corpora
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

import logging, gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# f = codecs.open('data/tweets.txt', 'r', "utf-8")
documents = codecs.open('data/tweets.txt', 'r', "utf-8")
lda_doc = codecs.open('data/tweets.txt', 'r', "utf-8")

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1).fit(X)

dbscan = DBSCAN(metric='euclidean').fit(X)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

print("\nPrediction")

Y = vectorizer.transform(["chrome."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat."])
prediction = model.predict(Y)
print(prediction)


stops = set(stopwords.words("english"))

#LDA
texts = [[word for word in document.lower().split() if word not in stops] for document in lda_doc]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, update_every=1, chunksize=10000, passes=10)
lda.print_topics(1)

documents.close()
lda_doc.close()
