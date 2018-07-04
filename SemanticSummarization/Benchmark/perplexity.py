import collections, nltk
from nltk import ngrams

corpus = """
Monty Python (sometimes known as The Pythons) were a British surreal comedy group who created the sketch comedy show Monty Python's Flying Circus,
that first aired on the BBC on October 5, 1969. Forty-five episodes were made over four series. The Python phenomenon developed from the television series
into something larger in scope and impact, spawning touring stage shows, films, numerous albums, several books, and a stage musical.
The group's influence on comedy has been compared to The Beatles' influence on music."""

# we first tokenize the text corpus
tokens = nltk.word_tokenize(corpus)


# here you construct the unigram language model
def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    for word in model:
        model[word] = model[word] / float(sum(model.values()))
    return model


# computes perplexity of the unigram model on a testset
def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1 / model[word])
    perplexity = pow(perplexity, 1 / float(N))
    return perplexity


testset1 = "Monty Python"
testset2 = "British surreal comedy group"
testset3 = "influence"
testset4 = "The Beatles"

model = unigram(tokens)
print(perplexity(testset1, model))
print(perplexity(testset2, model))
print(perplexity(testset3, model))
print(perplexity(testset4, model))
