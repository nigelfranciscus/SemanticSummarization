import pymongo
import tweepy

consumer_key = "QNOq44cKGVw07KMNFf9RZmpe5"
consumer_secret = "AJrlkQ5MKIyqg5o0U4dwWGQkWnUsiiaEqliGg1ybbtU2IBRXxW"
access_key = "352257626-eJG0KdSRjK4w29IrHorKKMPIyvgrjWBEGKIm1GvF"
access_secret = "YjvBHiN0LH4W6Y9bgCDNNuHS6yWizhxQr5a4Y3nKDkId0"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
tweepy.debug(True)


def process(get_tweets):
    for i in get_tweets:
        client = pymongo.MongoClient('localhost', 27017)
        db = client['twitter']
        collection = db['budget2018']
        collection.insert(i._json)


query = '#Budget2018'
location = "-27.963141,153.380654,2km"
max_tweets = 9999999999

total_tweets = 0
last_id = -1
while total_tweets < max_tweets:
    count = max_tweets - total_tweets
    try:
        # new_tweets = api.search(q="*", geocode=location, count=count, max_id=str(last_id - 1))
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))
        if not new_tweets:
            break
        last_id = new_tweets[-1].id
        process(new_tweets)
        total_tweets += len(new_tweets)
    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait
        # to keep things simple, we will give up on an error
        break
