
import pymongo
import codecs
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import ngrams
from nltk import pos_tag

from SemanticSummarization.Utils.twokenize import simpleTokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer as LsaRank
from sumy.summarizers.text_rank import TextRankSummarizer as TextRank
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRank

stops = set(stopwords.words("english"))
stemmer = PorterStemmer()


# tknzr = TweetTokenizer()
# tokens = word_tokenize(result)

# f = codecs.open('tweets.txt', 'w', "utf-8")

def mongo_connection(url, db_name, collection_name):
    final_result = set()

    connection = pymongo.MongoClient(url)[db_name]
    # collection = database.collection_names(include_system_collections=False)

    for data in connection[collection_name]. \
            find({}, {'_id': 0, 'text': 1, 'extended_tweet': 1, 'lang': 1, 'str_id': 1}).limit(1000):
        if "lang" not in data:
            continue
        if data["lang"] == "en":
            if 'extended_tweet' in data:
                data['text'] = data['extended_tweet']['full_text']

            result = data["text"].replace("\n", " ")
            result = result.replace("RT", "")
            result = re.sub(r"https\S+", "", result)
            result = re.sub('[^A-Za-z0-9]+', ' ', result).lower()

            # print("ID : %s \nOriginal Text: %s" % (data["id_str"], result))
            # output_tweet = (data["id_str"], result)
            # print(output_tweet[0])
            final_result.add(result.strip())

    return final_result


def pos_tag(text):
    tweet_token = simpleTokenize(text)
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


def sumy_summarizer(text, lang, sentence_count, sum_type):
    # url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files
    # TODO : Implement categories / paragraph vectoring beforehand.
    join_result = ".\n".join(text)
    parser = PlaintextParser.from_string(join_result, Tokenizer(lang))
    sumy_stemmer = Stemmer(lang)

    summarizer = sum_type(sumy_stemmer)
    summarizer.stop_words = get_stop_words(lang)

    for sentence in summarizer(parser.document, sentence_count):
        print(sentence)



uri = 'mongodb://bigdata:databig@localhost/?authSource=admin'
tweet = mongo_connection(uri, 'twitter', 'goldcoast_location')
# join_result = ". ".join(tweet)
# print(join_result)
# f = codecs.open('Data/goldcoast.txt', 'w', "utf-8")
# f.write(join_result)

sumy_summarizer(tweet, 'english', 10, TextRank)
print("\n")
sumy_summarizer(tweet, 'english', 10, LexRank)
print("\n")
sumy_summarizer(tweet, 'english', 10, LsaRank)
