import codecs

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score


# f = codecs.open('data/tweets.txt', 'r', "utf-8")
documents = codecs.open('data/tweets.txt', 'r', "utf-8")

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\nPrediction")

Y = vectorizer.transform(["chrome."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat."])
prediction = model.predict(Y)
print(prediction)

documents.close()