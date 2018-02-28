import pymongo
import codecs
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

    for data in database['goldcoast_location'].find({}, {'text': 1, '_id': 0, 'lang': 1}).sort("_id", -1).limit(100):
        if data['lang'] == 'en':
            result = data["text"].replace("\n", " ")
            result = re.sub(r"https\S+", "", result)
            result = result.lower()
            # print(result)
            print([i for i in result.split() if i not in stop])
            # f.write(result + "\n")
